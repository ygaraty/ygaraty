// Generated output: training_page.mjs.
import { contribute } from './training_client.mjs';
function element(id) {
    const value = document.getElementById(id);
    if (!value)
        throw Error(`Missing training page element: ${id}`);
    return value;
}
function button(id) {
    const value = element(id);
    if (!(value instanceof HTMLButtonElement))
        throw Error(`Expected button: ${id}`);
    return value;
}
const token = location.hash.slice(1);
history.replaceState(null, '', location.pathname);
const start = button('start'), stop = button('stop');
const state = element('state'), details = element('details');
let controller;
start.onclick = async () => {
    if (!token) {
        state.textContent = 'Open the full contributor link from the session owner.';
        return;
    }
    start.disabled = true;
    stop.disabled = false;
    controller = new AbortController();
    try {
        await contribute({ joinToken: token, signal: controller.signal, onStatus: value => {
                state.textContent = 'message' in value && value.message ? value.message : value.state;
                details.textContent = JSON.stringify(value, null, 2);
            } });
    }
    catch (error) {
        state.textContent = error instanceof Error ? error.message : String(error);
    }
    finally {
        start.disabled = false;
        stop.disabled = true;
    }
};
stop.onclick = () => { controller?.abort(); state.textContent = 'Stopped. The slot will become available after its lease expires.'; };
