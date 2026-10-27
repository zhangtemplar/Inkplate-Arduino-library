// Google Drive folder endpoint for Inkplate13SPECTRA_Digital_Picture_Frame.
//
// Deploy as a Google Apps Script web app:
// - Execute as: Me
// - Who has access: Anyone with the link
//
// Before deploying, run manifestOutput once in the editor and grant Drive
// permissions as the script owner. The final deployed /exec URL, not the
// editor-only /dev URL, must return JSON in a private browser window without
// any Google login or permission popup.
//
// The token below is checked by the script before listing or serving files.

const FOLDER_ID = 'PASTE_GOOGLE_DRIVE_FOLDER_ID_HERE';
const SYNC_TOKEN = 'CHANGE_THIS_SHARED_SECRET';

// Apps Script cannot run the Pillow-quality BIN converter directly. Run the
// companion Python converter against a local Google Drive synced folder, or use
// an external converter service, then let this manifest expose the BIN files.
const MANIFEST_BIN_ONLY = false;
const HIDE_SOURCE_IMAGES_WHEN_BIN_EXISTS = true;

const BIN_EXTENSION = '.bin';
const SOURCE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp'];
const MAX_MANIFEST_ITEMS = 48;

function doGet(e) {
  const params = e.parameter || {};
  if (params.token !== SYNC_TOKEN) {
    return jsonOutput({error: 'unauthorized'});
  }

  if (params.mode === 'file') {
    return fileOutput(params.id || '');
  }

  return manifestOutput();
}

function manifestOutput() {
  const folder = DriveApp.getFolderById(FOLDER_ID);
  const files = folder.getFiles();
  const candidates = [];
  const binStems = {};

  while (files.hasNext()) {
    const file = files.next();
    const name = file.getName();
    const extension = extensionForName(name);
    if (!isSupportedExtension(extension)) {
      continue;
    }

    const stem = filenameStem(name);
    candidates.push({
      file: file,
      name: name,
      extension: extension,
      stem: stem
    });

    if (extension === BIN_EXTENSION) {
      binStems[stem.toLowerCase()] = true;
    }
  }

  candidates.sort((a, b) => a.name.localeCompare(b.name));

  const photos = [];

  for (let i = 0; i < candidates.length && photos.length < MAX_MANIFEST_ITEMS; i++) {
    const item = candidates[i];
    if (!isManifestVisible(item, binStems)) {
      continue;
    }

    const file = item.file;
    photos.push({
      id: file.getId(),
      name: item.name,
      modified: file.getLastUpdated().toISOString(),
      size: file.getSize(),
      mimeType: file.getMimeType()
    });
  }

  return jsonOutput({photos: photos});
}

function fileOutput(id) {
  if (!id) {
    return jsonOutput({error: 'missing id'});
  }

  const file = DriveApp.getFileById(id);
  if (!isDownloadAllowedName(file.getName()) || !fileIsInConfiguredFolder(file)) {
    return jsonOutput({error: 'file not allowed'});
  }

  const bytes = file.getBlob().getBytes();
  const base64 = Utilities.base64Encode(bytes);
  return ContentService.createTextOutput(base64).setMimeType(ContentService.MimeType.TEXT);
}

function fileIsInConfiguredFolder(file) {
  const parents = file.getParents();
  while (parents.hasNext()) {
    if (parents.next().getId() === FOLDER_ID) {
      return true;
    }
  }
  return false;
}

function isSupportedName(name) {
  return isSupportedExtension(extensionForName(name));
}

function isSupportedExtension(extension) {
  return extension === BIN_EXTENSION || SOURCE_EXTENSIONS.indexOf(extension) >= 0;
}

function isDownloadAllowedName(name) {
  const extension = extensionForName(name);
  if (MANIFEST_BIN_ONLY) {
    return extension === BIN_EXTENSION;
  }
  return isSupportedExtension(extension);
}

function isManifestVisible(item, binStems) {
  if (MANIFEST_BIN_ONLY) {
    return item.extension === BIN_EXTENSION;
  }
  if (HIDE_SOURCE_IMAGES_WHEN_BIN_EXISTS && item.extension !== BIN_EXTENSION) {
    return !binStems[item.stem.toLowerCase()];
  }
  return true;
}

function extensionForName(name) {
  const lower = name.toLowerCase();
  const dot = lower.lastIndexOf('.');
  return dot >= 0 ? lower.substring(dot) : '';
}

function filenameStem(name) {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.substring(0, dot) : name;
}

function jsonOutput(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}
