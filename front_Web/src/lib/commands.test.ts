import { describe, it, expect } from 'vitest';
import { parseSlashCommand } from './commands';

describe('parseSlashCommand', () => {
  it('parses help', () => {
    expect(parseSlashCommand('/help')).toEqual({ type: 'help' });
  });

  it('parses switch with id', () => {
    expect(parseSlashCommand('/switch abc-123')).toEqual({
      type: 'switch',
      threadId: 'abc-123',
    });
  });

  it('returns message for plain text', () => {
    expect(parseSlashCommand('头疼')).toEqual({ type: 'message', text: '头疼' });
  });
});
