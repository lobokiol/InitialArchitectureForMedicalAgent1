import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Trash2, Bot, User, Loader2, MessageSquare, Plus, Settings, HelpCircle } from 'lucide-react';
import { useChat } from '../hooks/useChat';
import { useUser } from '../hooks/useUser';
import { useThreads } from '../hooks/useThreads';
import { UserModal } from './UserModal';
import { ThreadList } from './ThreadList';
import { CommandHelp } from './CommandHelp';
import { IntentBadge } from './IntentBadge';

export function LiveDemo() {
  const { user, isLoading: userLoading, showModal, login, logout, openModal, closeModal } = useUser();
  const {
    threads,
    currentThread,
    isLoading: threadLoading,
    showThreadList,
    createNewThread,
    switchToThread,
    removeThread,
    openThreadList,
    closeThreadList,
  } = useThreads(user?.user_id || null);
  
  const {
    messages,
    isLoading: chatLoading,
    error: chatError,
    currentThreadId,
    sendMessage,
    clearMessages,
    messagesEndRef,
  } = useChat(user?.user_id || null, currentThread?.thread_id || null);

  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const [showHelp, setShowHelp] = useState(false);
  const isProcessing = chatLoading || threadLoading;

  useEffect(() => {
    if (!showModal && user && !currentThread) {
      openThreadList();
    }
  }, [showModal, user, currentThread, openThreadList]);

  const handleCommand = async (cmd: string) => {
    const command = cmd.trim().toLowerCase();
    
    switch (command) {
      case '/help':
        setShowHelp(true);
        break;
      case '/new':
        await createNewThread();
        clearMessages();
        break;
      case '/threads':
        openThreadList();
        break;
      case '/switch':
        openThreadList();
        break;
      case '/delete':
        if (currentThread?.thread_id) {
          await removeThread(currentThread.thread_id);
          clearMessages();
        }
        break;
      case '/user':
        openModal();
        break;
      default:
        return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing || !user) return;

    if (input.trim().startsWith('/')) {
      const handled = await handleCommand(input);
      if (handled) {
        setInput('');
      }
      return;
    }

    await sendMessage(input);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  return (
    <section id="demo" className="py-20 bg-gradient-to-br from-primary-50 to-medical-50">
      <div className="max-w-4xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-4">实时演示</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            连接后端 API，体验智能导诊对话（请确保后端服务已启动）
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100"
        >
          {/* 头部 */}
          <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-100">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="flex items-center gap-2 text-sm">
                {user && (
                  <button
                    onClick={openModal}
                    className="flex items-center gap-1 px-2 py-1 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <User className="w-3.5 h-3.5 text-gray-500" />
                    <span className="font-medium text-gray-700">
                      {user.name || user.user_id}
                    </span>
                    <Settings className="w-3 h-3 text-gray-400" />
                  </button>
                )}
                <span className="text-gray-300">|</span>
                <button
                  onClick={openThreadList}
                  className="flex items-center gap-1 px-2 py-1 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <MessageSquare className="w-3.5 h-3.5 text-gray-500" />
                  <span className="text-gray-600 truncate max-w-[150px]">
                    {currentThread?.title || '选择会话'}
                  </span>
                </button>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setShowHelp(true)}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                title="帮助"
              >
                <HelpCircle className="w-4 h-4" />
              </button>
              <button
                onClick={clearMessages}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                title="清空对话"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* 消息区域 */}
          <div className="h-[400px] overflow-y-auto p-4 space-y-4">
            {!user ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <User className="w-12 h-12 mb-3 opacity-50" />
                <p className="text-center">
                  请先设置用户信息<br />
                  <button
                    onClick={openModal}
                    className="text-primary-600 hover:underline mt-2"
                  >
                    点击设置
                  </button>
                </p>
              </div>
            ) : messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <Bot className="w-12 h-12 mb-3 opacity-50" />
                <p className="text-center">
                  你好！我是医院导诊助手<br />
                  <span className="text-sm">可以问我关于挂号、科室、就医流程等问题</span>
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-2 text-xs">
                  <button
                    onClick={() => setShowHelp(true)}
                    className="px-2 py-1 bg-gray-100 rounded hover:bg-gray-200"
                  >
                    /help 查看命令
                  </button>
                  <button
                    onClick={openThreadList}
                    className="px-2 py-1 bg-gray-100 rounded hover:bg-gray-200"
                  >
                    /new 新建会话
                  </button>
                </div>
              </div>
            ) : (
              <AnimatePresence>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.role === 'user' ? 'bg-gray-100' : 'bg-primary-600'
                    }`}>
                      {message.role === 'user' ? (
                        <User className="w-4 h-4 text-gray-600" />
                      ) : (
                        <Bot className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <div className={`max-w-[70%] ${message.role === 'user' ? 'text-right' : ''}`}>
                      <div className={`inline-block max-w-full rounded-2xl px-4 py-3 ${
                        message.role === 'user' 
                          ? 'bg-primary-100 text-gray-900' 
                          : 'bg-gray-50 text-gray-900'
                      }`}>
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                      </div>
                      {message.role === 'assistant' && (
                        <IntentBadge
                          intent={message.intent_result}
                          usedDocs={message.used_docs}
                        />
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}

            {chatLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-3"
              >
                <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-gray-50 rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-2 text-gray-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">正在思考...</span>
                  </div>
                </div>
              </motion.div>
            )}

            {chatError && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-red-600 text-sm"
              >
                {chatError}
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* 输入区域 */}
          <form onSubmit={handleSubmit} className="border-t border-gray-100 p-4">
            <div className="flex gap-3">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={user ? "输入问题或命令（如 /help）..." : "请先设置用户信息"}
                className="flex-1 px-4 py-3 rounded-xl border border-gray-200 focus:border-primary-300 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-gray-900 placeholder-gray-400"
                disabled={!user || isProcessing}
              />
              <button
                type="submit"
                disabled={!input.trim() || !user || isProcessing}
                className="px-6 py-3 bg-primary-600 text-white font-medium rounded-xl hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                <Send className="w-4 h-4" />
                <span className="hidden sm:inline">发送</span>
              </button>
            </div>
          </form>
        </motion.div>

        {user && (
          <p className="text-center text-sm text-gray-500 mt-4">
            示例问题：<span className="text-primary-600 cursor-pointer hover:underline" onClick={() => sendMessage('我头疼应该挂什么科？')}>"我头疼应该挂什么科？"</span>
            <span className="mx-2">|</span>
            <span className="text-primary-600 cursor-pointer hover:underline" onClick={() => sendMessage('感冒了怎么看医生？')}>"感冒了怎么看医生？"</span>
            <span className="mx-2">|</span>
            <span className="text-primary-600 cursor-pointer hover:underline" onClick={() => sendMessage('取药需要什么流程？')}>"取药需要什么流程？"</span>
          </p>
        )}
      </div>

      <UserModal
        isOpen={showModal}
        onClose={closeModal}
        onLogin={login}
        isLoading={userLoading}
      />

      <ThreadList
        isOpen={showThreadList}
        onClose={closeThreadList}
        threads={threads}
        currentThreadId={currentThread?.thread_id}
        onSwitch={switchToThread}
        onDelete={removeThread}
        onCreate={createNewThread}
        isLoading={threadLoading}
      />

      <CommandHelp
        isOpen={showHelp}
        onClose={() => setShowHelp(false)}
      />
    </section>
  );
}
