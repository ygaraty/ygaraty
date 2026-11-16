// Slot-scoped versions isolate reset/cancellation from other active chats.
export class ChatSlots {
  constructor(count) {
    this.epochs = Array(count).fill(0);
    this.turns = Array(count).fill(0);
  }
  admit(message, held) {
    const {kind, meta} = message, slot = meta.conv ?? 0;
    if (!Number.isInteger(slot) || slot < 0 || slot >= this.epochs.length) return false;
    if (!Number.isSafeInteger(meta.chat_epoch) || meta.chat_epoch < this.epochs[slot]) return false;
    if (kind === 'reset') {
      this.epochs[slot] = meta.chat_epoch;
      this.turns[slot] = 0;
      return true;
    }
    if (meta.chat_epoch > this.epochs[slot]) { held.push(message); return false; }
    if (!Number.isSafeInteger(meta.chat_turn) || meta.chat_turn < this.turns[slot]) return false;
    this.turns[slot] = meta.chat_turn;
    return true;
  }
  stamp(meta) {
    const slot = meta.conv ?? 0;
    return {...meta, chat_epoch:this.epochs[slot], chat_turn:this.turns[slot]};
  }
  clear() { this.epochs.fill(0); this.turns.fill(0); }
}
