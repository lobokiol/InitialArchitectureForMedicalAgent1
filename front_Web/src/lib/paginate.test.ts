import { describe, it, expect } from 'vitest';
import { paginateArray, paginateText } from './paginate';

describe('paginateArray', () => {
  it('splits into pages', () => {
    expect(paginateArray([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it('returns empty page for empty input', () => {
    expect(paginateArray([], 3)).toEqual([[]]);
  });
});

describe('paginateText', () => {
  it('splits long text', () => {
    const pages = paginateText('abcdefgh', 3);
    expect(pages).toEqual(['abc', 'def', 'gh']);
  });
});
