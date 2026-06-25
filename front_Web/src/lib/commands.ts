import type { SlashCommand } from '../types';

export function parseSlashCommand(input: string): SlashCommand {
  const t = input.trim();
  if (!t.startsWith('/')) return { type: 'message', text: t };
  const [cmd, ...rest] = t.slice(1).split(/\s+/);
  const arg = rest.join(' ');
  switch (cmd.toLowerCase()) {
    case 'help':
      return { type: 'help' };
    case 'new':
      return { type: 'new' };
    case 'threads':
      return { type: 'threads' };
    case 'switch':
      return { type: 'switch', threadId: arg };
    case 'delete':
      return { type: 'delete' };
    case 'user':
      return { type: 'user' };
    case 'exit':
      return { type: 'exit' };
    default:
      return { type: 'message', text: t };
  }
}
