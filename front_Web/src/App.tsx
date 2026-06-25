import { useCallback, useEffect, useState } from 'react';
import { AppShell } from './components/AppShell';
import { ChatInput } from './components/ChatInput';
import { ChatStage } from './components/ChatStage';
import { CommandHelp } from './components/CommandHelp';
import { DetailOverlay } from './components/DetailOverlay';
import { HelpModal } from './components/HelpModal';
import { InfoBar } from './components/InfoBar';
import { SettingsPanel } from './components/SettingsPanel';
import { ThreadSidebar } from './components/ThreadSidebar';
import { Toast } from './components/Toast';
import { TopBar } from './components/TopBar';
import { UserModal } from './components/UserModal';
import { useChat } from './hooks/useChat';
import { useReadiness } from './hooks/useReadiness';
import { useThreads } from './hooks/useThreads';
import { useUser } from './hooks/useUser';
import type { SlashCommand } from './types';

export default function App() {
  const user = useUser();
  const threads = useThreads(user.userId);
  const readiness = useReadiness();
  const chat = useChat(user.userId, threads.currentThreadId);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [commandHelpOpen, setCommandHelpOpen] = useState(false);
  const [userModalOpen, setUserModalOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [fullText, setFullText] = useState<string | undefined>();
  const [toast, setToast] = useState<string | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 5000);
  }, []);

  useEffect(() => {
    if (chat.error) showToast(chat.error);
  }, [chat.error, showToast]);

  useEffect(() => {
    if (threads.error) showToast(threads.error);
  }, [threads.error, showToast]);

  const handleUserSubmit = async (userId: string, name: string) => {
    const result = await user.initUser(userId, name || undefined);
    setUserModalOpen(false);
    await threads.refresh(result.userId);
    if (result.degraded) {
      showToast(`用户已本地保存（/users 异常: ${result.error}）`);
    }
  };

  const handleSelectThread = async (threadId: string) => {
    await threads.switchTo(threadId);
    chat.clearTurns();
    setDrawerOpen(false);
  };

  const handleNewThread = async () => {
    await threads.createNew();
    chat.clearTurns();
    setDrawerOpen(false);
  };

  const handleDeleteThread = async (threadId: string) => {
    await threads.remove(threadId);
    if (threadId === threads.currentThreadId) chat.clearTurns();
  };

  const handleCommand = useCallback(
    async (cmd: SlashCommand) => {
      if (!cmd) return;
      switch (cmd.type) {
        case 'help':
          setCommandHelpOpen(true);
          break;
        case 'new':
          await handleNewThread();
          break;
        case 'threads':
          setDrawerOpen(true);
          break;
        case 'switch':
          if (cmd.threadId) {
            await handleSelectThread(cmd.threadId);
          } else showToast('用法: /switch <thread_id>');
          break;
        case 'delete':
          if (threads.currentThreadId) await handleDeleteThread(threads.currentThreadId);
          break;
        case 'user':
          setUserModalOpen(true);
          break;
        case 'exit':
          setSettingsOpen(false);
          setHelpOpen(false);
          setCommandHelpOpen(false);
          break;
        default:
          break;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [threads.currentThreadId],
  );

  const handleSend = async (text: string) => {
    const cmd = await chat.sendMessage(text);
    if (cmd) await handleCommand(cmd);
  };

  const handlePickChoice = async (message: string) => {
    await chat.pickChoice(message);
  };

  const sidebarProps = {
    threads: threads.threads,
    currentThreadId: threads.currentThreadId,
    loading: threads.loading,
    onSelect: handleSelectThread,
    onNew: handleNewThread,
    onDelete: handleDeleteThread,
  };

  const inputDisabled = chat.phase === 'loading' || !user.userId;

  return (
    <>
      <UserModal
        open={user.needsOnboarding || userModalOpen}
        forced={user.needsOnboarding}
        loading={user.loading}
        onSubmit={handleUserSubmit}
        onClose={() => setUserModalOpen(false)}
      />

      <AppShell
        drawerOpen={drawerOpen}
        onCloseDrawer={() => setDrawerOpen(false)}
        topBar={
          <TopBar
            userId={user.userId}
            threadTitle={threads.currentTitle}
            dotColor={readiness.dotColor}
            readinessLabel={readiness.label}
            onOpenDrawer={() => setDrawerOpen(true)}
            onOpenHelp={() => setHelpOpen(true)}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        }
        sidebar={<ThreadSidebar {...sidebarProps} />}
        mobileDrawer={<ThreadSidebar {...sidebarProps} />}
        main={
          <>
            <ChatStage
              turn={chat.currentTurn}
              viewIndex={chat.viewIndex}
              totalTurns={chat.turns.length}
              phase={chat.phase}
              onPrev={chat.goPrev}
              onNext={chat.goNext}
              onPickChoice={handlePickChoice}
              onExpandFull={(text) => {
                setFullText(text);
                setDetailOpen(true);
              }}
            />
            <ChatInput disabled={inputDisabled} onSend={handleSend} />
            <InfoBar
              response={chat.lastResponse}
              onOpenDetail={() => {
                setFullText(undefined);
                setDetailOpen(true);
              }}
            />
          </>
        }
      />

      <DetailOverlay
        open={detailOpen}
        response={chat.lastResponse}
        fullText={fullText}
        onClose={() => {
          setDetailOpen(false);
          setFullText(undefined);
        }}
      />
      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
      <SettingsPanel
        open={settingsOpen}
        ready={readiness.ready}
        apiError={readiness.error}
        userId={user.userId}
        userName={user.userName}
        onClose={() => setSettingsOpen(false)}
        onEditUser={() => {
          setSettingsOpen(false);
          setUserModalOpen(true);
        }}
      />
      <CommandHelp open={commandHelpOpen} onClose={() => setCommandHelpOpen(false)} />
      <Toast message={toast} onDismiss={() => setToast(null)} />
    </>
  );
}
