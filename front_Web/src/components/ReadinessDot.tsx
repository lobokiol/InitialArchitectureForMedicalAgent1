interface ReadinessDotProps {
  color: string;
  label: string;
}

export function ReadinessDot({ color, label }: ReadinessDotProps) {
  return (
    <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-brand-50 text-xs text-gray-600">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      <span className="hidden sm:inline">{label}</span>
    </div>
  );
}
