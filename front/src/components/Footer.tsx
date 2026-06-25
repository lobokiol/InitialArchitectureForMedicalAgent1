import { Github, Mail } from 'lucide-react';

export function Footer() {
  return (
    <footer className="py-8 bg-gray-900 text-gray-400">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-center md:text-left">
            <h3 className="text-white font-semibold mb-1">医院导诊 Agentic 助手</h3>
            <p className="text-sm">基于 FastAPI + LangGraph 的生产级智能导诊系统</p>
          </div>
          
          <div className="flex items-center gap-4">
            <a
              href="#"
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              title="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="#"
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              title="联系"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>
        
        <div className="mt-6 pt-6 border-t border-gray-800 text-center text-sm">
          <p>© 2024 医院导诊系统. 技术演示项目.</p>
        </div>
      </div>
    </footer>
  );
}
