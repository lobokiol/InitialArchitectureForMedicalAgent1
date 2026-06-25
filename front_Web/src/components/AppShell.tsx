import type { ReactNode } from 'react';

interface AppShellProps {
  topBar: ReactNode;
  sidebar: ReactNode;
  main: ReactNode;
  mobileDrawer: ReactNode;
  drawerOpen: boolean;
  onCloseDrawer: () => void;
}

export function AppShell({
  topBar,
  sidebar,
  main,
  mobileDrawer,
  drawerOpen,
  onCloseDrawer,
}: AppShellProps) {
  return (
    <div className="h-screen overflow-hidden flex flex-col bg-brand-50">
      {topBar}
      <div className="flex flex-1 min-h-0 relative">
        <aside className="hidden md:flex w-60 shrink-0 border-r border-brand-500/20 bg-white flex-col min-h-0 overflow-hidden">
          {sidebar}
        </aside>
        <main className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden">{main}</main>
        {drawerOpen && (
          <div className="md:hidden fixed inset-0 z-40 flex">
            <button
              type="button"
              className="flex-1 bg-black/30"
              aria-label="关闭会话列表"
              onClick={onCloseDrawer}
            />
            <div className="w-72 max-w-[85vw] bg-white shadow-xl flex flex-col min-h-0 overflow-hidden">
              {mobileDrawer}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
