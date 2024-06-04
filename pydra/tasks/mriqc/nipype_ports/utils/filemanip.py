from fileformats.generic import File
from hashlib import md5
import hashlib
import logging
import os
import os.path as op
import posixpath
import re
import shutil
import subprocess as sp


logger = logging.getLogger(__name__)


def _generate_cifs_table():
    """Construct a reverse-length-ordered list of mount points that
    fall under a CIFS mount.

    This precomputation allows efficient checking for whether a given path
    would be on a CIFS filesystem.

    On systems without a ``mount`` command, or with no CIFS mounts, returns an
    empty list.
    """
    exit_code, output = sp.getstatusoutput("mount")
    return _parse_mount_table(exit_code, output)


def _parse_mount_table(exit_code, output):
    """Parses the output of ``mount`` to produce (path, fs_type) pairs

    Separated from _generate_cifs_table to enable testing logic with real
    outputs
    """
    # Not POSIX
    if exit_code != 0:
        return []
    # Linux mount example:  sysfs on /sys type sysfs (rw,nosuid,nodev,noexec)
    # <PATH>^^^^      ^^^^^<FSTYPE>
    # OSX mount example:    /dev/disk2 on / (hfs, local, journaled)
    # <PATH>^  ^^^<FSTYPE>
    pattern = re.compile(r".*? on (/.*?) (?:type |\()([^\s,\)]+)")
    # Keep line and match for error reporting (match == None on failure)
    # Ignore empty lines
    matches = [(l, pattern.match(l)) for l in output.strip().splitlines() if l]
    # (path, fstype) tuples, sorted by path length (longest first)
    mount_info = sorted(
        (match.groups() for _, match in matches if match is not None),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    cifs_paths = [path for path, fstype in mount_info if fstype.lower() == "cifs"]
    # Report failures as warnings
    for line, match in matches:
        if match is None:
            fmlogger.debug("Cannot parse mount line: '%s'", line)
    return [
        mount
        for mount in mount_info
        if any(mount[0].startswith(path) for path in cifs_paths)
    ]


def copyfile(
    originalfile,
    newfile,
    copy=False,
    create_new=False,
    hashmethod=None,
    use_hardlink=False,
    copy_related_files=True,
):
    """Copy or link ``originalfile`` to ``newfile``.

    If ``use_hardlink`` is True, and the file can be hard-linked, then a
    link is created, instead of copying the file.

    If a hard link is not created and ``copy`` is False, then a symbolic
    link is created.

    Parameters
    ----------
    originalfile : str
        full path to original file
    newfile : str
        full path to new file
    copy : Bool
        specifies whether to copy or symlink files
        (default=False) but only for POSIX systems
    use_hardlink : Bool
        specifies whether to hard-link files, when able
        (Default=False), taking precedence over copy
    copy_related_files : Bool
        specifies whether to also operate on related files, as defined in
        ``related_filetype_sets``

    Returns
    -------
    None

    """
    newhash = None
    orighash = None
    fmlogger.debug(newfile)
    if create_new:
        while op.exists(newfile):
            base, fname, ext = split_filename(newfile)
            s = re.search("_c[0-9]{4,4}$", fname)
            i = 0
            if s:
                i = int(s.group()[2:]) + 1
                fname = fname[:-6] + "_c%04d" % i
            else:
                fname += "_c%04d" % i
            newfile = base + os.sep + fname + ext
    if hashmethod is None:
        hashmethod = config.get("execution", "hash_method").lower()
    # Don't try creating symlinks on CIFS
    if copy is False and on_cifs(newfile):
        copy = True
    # Existing file
    # -------------
    # Options:
    # symlink
    # to regular file originalfile            (keep if symlinking)
    # to same dest as symlink originalfile    (keep if symlinking)
    # to other file                           (unlink)
    # regular file
    # hard link to originalfile               (keep)
    # copy of file (same hash)                (keep)
    # different file (diff hash)              (unlink)
    keep = False
    if op.lexists(newfile):
        if op.islink(newfile):
            if all(
                (
                    os.readlink(newfile) == op.realpath(originalfile),
                    not use_hardlink,
                    not copy,
                )
            ):
                keep = True
        elif posixpath.samefile(newfile, originalfile):
            keep = True
        else:
            if hashmethod == "timestamp":
                hashfn = hash_timestamp
            elif hashmethod == "content":
                hashfn = hash_infile
            else:
                raise AttributeError("Unknown hash method found:", hashmethod)
            newhash = hashfn(newfile)
            fmlogger.debug(
                "File: %s already exists,%s, copy:%d", newfile, newhash, copy
            )
            orighash = hashfn(originalfile)
            keep = newhash == orighash
        if keep:
            fmlogger.debug(
                "File: %s already exists, not overwriting, copy:%d", newfile, copy
            )
        else:
            os.unlink(newfile)
    # New file
    # --------
    # use_hardlink & can_hardlink => hardlink
    # ~hardlink & ~copy & can_symlink => symlink
    # ~hardlink & ~symlink => copy
    if not keep and use_hardlink:
        try:
            fmlogger.debug("Linking File: %s->%s", newfile, originalfile)
            # Use realpath to avoid hardlinking symlinks
            os.link(op.realpath(originalfile), newfile)
        except OSError:
            use_hardlink = False  # Disable hardlink for associated files
        else:
            keep = True
    if not keep and not copy and os.name == "posix":
        try:
            fmlogger.debug("Symlinking File: %s->%s", newfile, originalfile)
            os.symlink(originalfile, newfile)
        except OSError:
            copy = True  # Disable symlink for associated files
        else:
            keep = True
    if not keep:
        try:
            fmlogger.debug("Copying File: %s->%s", newfile, originalfile)
            shutil.copyfile(originalfile, newfile)
        except shutil.Error as e:
            fmlogger.warning(str(e))
    # Associated files
    if copy_related_files:
        related_file_pairs = (
            get_related_files(f, include_this_file=False)
            for f in (originalfile, newfile)
        )
        for alt_ofile, alt_nfile in zip(*related_file_pairs):
            if op.exists(alt_ofile):
                copyfile(
                    alt_ofile,
                    alt_nfile,
                    copy,
                    hashmethod=hashmethod,
                    use_hardlink=use_hardlink,
                    copy_related_files=False,
                )
    return newfile


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import type(attrs.NOTHING)
    >>> fname_presuffix(fname, 'pre', 'post', type(attrs.NOTHING)) ==              fname_presuffix(fname, 'pre', 'post')
    True

    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""
    # No need for : bool(type(attrs.NOTHING) is not attrs.NOTHING) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return op.join(pth, prefix + fname + suffix + ext)


def get_related_files(filename, include_this_file=True):
    """Returns a list of related files, as defined in
    ``related_filetype_sets``, for a filename. (e.g., Nifti-Pair, Analyze (SPM)
    and AFNI files).

    Parameters
    ----------
    filename : str
        File name to find related filetypes of.
    include_this_file : bool
        If true, output includes the input filename.
    """
    related_files = []
    path, name, this_type = split_filename(filename)
    for type_set in related_filetype_sets:
        if this_type in type_set:
            for related_type in type_set:
                if include_this_file or related_type != this_type:
                    related_files.append(op.join(path, name + related_type))
    if not len(related_files):
        related_files = [filename]
    return related_files


def hash_infile(afile, chunk_len=8192, crypto=hashlib.md5, raise_notfound=False):
    """
    Computes hash of a file using 'crypto' module

    >>> hash_infile('smri_ants_registration_settings.json')
    'f225785dfb0db9032aa5a0e4f2c730ad'

    >>> hash_infile('surf01.vtk')
    'fdf1cf359b4e346034372cdeb58f9a88'

    >>> hash_infile('spminfo')
    '0dc55e3888c98a182dab179b976dfffc'

    >>> hash_infile('fsl_motion_outliers_fd.txt')
    'defd1812c22405b1ee4431aac5bbdd73'


    """
    if not op.isfile(afile):
        if raise_notfound:
            raise RuntimeError('File "%s" not found.' % afile)
        return None
    crypto_obj = crypto()
    with open(afile, "rb") as fp:
        while True:
            data = fp.read(chunk_len)
            if not data:
                break
            crypto_obj.update(data)
    return crypto_obj.hexdigest()


def hash_timestamp(afile):
    """Computes md5 hash of the timestamp of a file"""
    md5hex = None
    if op.isfile(afile):
        md5obj = md5()
        stat = os.stat(afile)
        md5obj.update(str(stat.st_size).encode())
        md5obj.update(str(stat.st_mtime).encode())
        md5hex = md5obj.hexdigest()
    return md5hex


def on_cifs(fname):
    """
    Checks whether a file path is on a CIFS filesystem mounted in a POSIX
    host (i.e., has the ``mount`` command).

    On Windows, Docker mounts host directories into containers through CIFS
    shares, which has support for Minshall+French symlinks, or text files that
    the CIFS driver exposes to the OS as symlinks.
    We have found that under concurrent access to the filesystem, this feature
    can result in failures to create or read recently-created symlinks,
    leading to inconsistent behavior and ``FileNotFoundError``.

    This check is written to support disabling symlinks on CIFS shares.

    """
    # Only the first match (most recent parent) counts
    for fspath, fstype in _cifs_table:
        if fname.startswith(fspath):
            return fstype == "cifs"
    return False


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """
    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]
    pth = op.dirname(fname)
    fname = op.basename(fname)
    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)
    return pth, fname, ext


_cifs_table = _generate_cifs_table()

fmlogger = logging.getLogger("nipype.utils")

related_filetype_sets = [(".hdr", ".img", ".mat"), (".nii", ".mat"), (".BRIK", ".HEAD")]
