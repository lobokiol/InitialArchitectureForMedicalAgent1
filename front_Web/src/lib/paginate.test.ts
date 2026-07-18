import { describe, it, expect } from 'vitest';
import { paginateArray } from './paginate';

describe('paginateArray', () => {
  it('splits into pages', () => {
    expect(paginateArray([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it('returns empty page for empty input', () => {
    expect(paginateArray([], 3)).toEqual([[]]);
  });
});
