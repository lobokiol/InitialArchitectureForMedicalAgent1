import type { OnCallDoctor } from '../types';

interface AppointmentCardsProps {
  doctors: OnCallDoctor[];
}

export function AppointmentCards({ doctors }: AppointmentCardsProps) {
  if (!doctors.length) return null;

  return (
    <div className="w-full max-w-lg mx-auto">
      <p className="text-xs font-medium text-brand-700 mb-2">值班医生预约</p>
      <div className="flex flex-col sm:flex-row gap-2">
        {doctors.map((doc) => {
          const full = doc.slots <= 0;
          return (
            <div
              key={doc.name}
              className="flex-1 rounded-xl border border-brand-500/20 bg-white px-3 py-3 shadow-sm"
            >
              <p className="font-medium text-gray-900 text-sm">{doc.name}</p>
              <p className="text-xs text-gray-500 mt-1">{doc.time}</p>
              <p className="text-xs text-brand-700 mt-1">{full ? '已满' : `余号 ${doc.slots}`}</p>
              <button
                type="button"
                disabled
                className="mt-2 w-full rounded-lg bg-brand-500/90 text-white text-xs py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                预约
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
