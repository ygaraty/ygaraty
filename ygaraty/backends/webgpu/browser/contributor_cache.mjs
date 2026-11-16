// "Saved model files" on the contributor page: how much this browser keeps
// for this server, and a button that deletes it. Weights are cached by hash
// in Cache Storage (qwen35_contributor.mjs) so a reload skips the download;
// clearing only means the next load downloads them again. Weights already on
// the GPU keep working until the tab closes.
export const CACHE_NAMES = ['ygaraty-qwen35'];

function gigabytes(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

export async function summary({caches: store = globalThis.caches, storage = globalThis.navigator?.storage} = {}) {
  if (!store) return {available: false, files: 0, bytes: null};
  let files = 0;
  for (const name of CACHE_NAMES) {
    if (!(await store.has(name))) continue;
    files += (await (await store.open(name)).keys()).length;
  }
  // Usage of this whole site in this browser; the weights are nearly all of it.
  let bytes = null;
  try { bytes = (await storage?.estimate?.())?.usage ?? null; } catch {}
  return {available: true, files, bytes};
}

export function describe({available, files, bytes}) {
  if (!available) return 'This browser does not keep model files for this page.';
  if (!files) return 'No model files saved in this browser.';
  const size = bytes ? ` (${gigabytes(bytes)})` : '';
  return `${files} model file${files === 1 ? '' : 's'} saved in this browser${size}, so reloading skips the download.`;
}

export async function clear({caches: store = globalThis.caches} = {}) {
  if (!store) return 0;
  let removed = 0;
  for (const name of CACHE_NAMES) if (await store.delete(name)) removed += 1;
  return removed;
}

export function mount({element, button, confirm, ...apis}) {
  if (!element || !button) return null;
  let deleting = false;
  const refresh = async () => {
    if (deleting) return;
    try {
      const state = await summary(apis);
      element.textContent = describe(state);
      button.disabled = !state.files;
    } catch (error) {
      element.textContent = `Could not read saved model files: ${error.message}`;
      button.disabled = true;
    }
  };
  button.onclick = async () => {
    const ask = confirm ?? globalThis.confirm;
    if (ask && !ask('Delete the model files saved in this browser? The next load downloads them again; a model already running in this tab keeps working.')) return;
    button.disabled = true;
    deleting = true;
    element.textContent = 'Deleting saved model files…';
    try {
      await clear(apis);
    } catch (error) {
      element.textContent = `Could not delete: ${error.message}`;
      return;
    } finally {
      deleting = false;
    }
    await refresh();
  };
  refresh();
  return refresh;
}

const element = globalThis.document?.getElementById?.('saved-files');
if (element) {
  const refresh = mount({element, button: document.getElementById('clear-saved-files')});
  // Downloads fill the cache while the model loads: keep the count current.
  setInterval(() => { if (document.visibilityState === 'visible') refresh(); }, 5000);
}
