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

const SUPPORTED_EXTENSIONS = ['.bin', '.jpg', '.jpeg', '.png', '.bmp'];
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
  const photos = [];

  while (files.hasNext() && photos.length < MAX_MANIFEST_ITEMS) {
    const file = files.next();
    const name = file.getName();
    if (!isSupportedName(name)) {
      continue;
    }

    photos.push({
      id: file.getId(),
      name: name,
      modified: file.getLastUpdated().toISOString(),
      size: file.getSize(),
      mimeType: file.getMimeType()
    });
  }

  photos.sort((a, b) => a.name.localeCompare(b.name));
  return jsonOutput({photos: photos});
}

function fileOutput(id) {
  if (!id) {
    return jsonOutput({error: 'missing id'});
  }

  const file = DriveApp.getFileById(id);
  if (!isSupportedName(file.getName()) || !fileIsInConfiguredFolder(file)) {
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
  const lower = name.toLowerCase();
  return SUPPORTED_EXTENSIONS.some(ext => lower.endsWith(ext));
}

function jsonOutput(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}
