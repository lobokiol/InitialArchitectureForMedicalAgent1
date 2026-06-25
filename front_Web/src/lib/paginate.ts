export function paginateArray<T>(items: T[], pageSize: number): T[][] {
  if (pageSize <= 0) return [items];
  const pages: T[][] = [];
  for (let i = 0; i < items.length; i += pageSize) pages.push(items.slice(i, i + pageSize));
  return pages.length ? pages : [[]];
}

export function paginateText(text: string, maxChars: number): string[] {
  if (!text) return [''];
  if (text.length <= maxChars) return [text];
  const pages: string[] = [];
  for (let i = 0; i < text.length; i += maxChars) pages.push(text.slice(i, i + maxChars));
  return pages;
}
