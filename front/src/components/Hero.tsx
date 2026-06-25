import { motion } from 'framer-motion';
import { Stethoscope, ArrowDown } from 'lucide-react';

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.8, ease: 'easeOut' }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.15
    }
  }
};

export function Hero() {
  const scrollToDemo = () => {
    const demoSection = document.getElementById('demo');
    demoSection?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-white to-medical-50 relative overflow-hidden">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMtOS45NDEgMC0xOCA4LjA1OS0xOCAxOHM4LjA1OSAxOCAxOCAxOCAxOC04LjA1OSAxOC0xOC04LjA1OS0xOC0xOC0xOHptMCAzMmMtNy43MzIgMC0xNC02LjI2OC0xNC0xNHM2LjI2OC0xNCAxNC0xNCAxNCA2LjI2OCAxNCAxNC02LjI2OCAxNC0xNCAxNHoiIGZpbGw9IiMxZTgwYWIiIGZpbGwtb3BhY2l0eT0iLjAyIi8+PC9nPjwvc3ZnPg==')] opacity-30"></div>
      
      <motion.div 
        className="text-center px-4 relative z-10"
        variants={staggerContainer}
        initial="initial"
        animate="animate"
      >
        <motion.div 
          variants={fadeInUp}
          className="flex items-center justify-center gap-3 mb-6"
        >
          <div className="p-4 bg-primary-600 rounded-2xl shadow-lg">
            <Stethoscope className="w-10 h-10 text-white" />
          </div>
        </motion.div>

        <motion.h1 
          variants={fadeInUp}
          className="text-5xl md:text-6xl font-bold text-gray-900 mb-4"
        >
          医院导诊 <span className="text-primary-600">Agentic</span> 助手
        </motion.h1>

        <motion.p 
          variants={fadeInUp}
          className="text-xl md:text-2xl text-gray-600 mb-8 max-w-2xl mx-auto"
        >
          基于 LangGraph 的智能导诊问答系统
        </motion.p>

        <motion.div 
          variants={fadeInUp}
          className="flex flex-wrap justify-center gap-4 mb-12"
        >
          <span className="px-4 py-2 bg-white rounded-full text-sm font-medium text-gray-700 shadow-sm">
            🤖 Agentic 对话
          </span>
          <span className="px-4 py-2 bg-white rounded-full text-sm font-medium text-gray-700 shadow-sm">
            🔍 RAG 检索
          </span>
          <span className="px-4 py-2 bg-white rounded-full text-sm font-medium text-gray-700 shadow-sm">
            🏥 症状问诊
          </span>
        </motion.div>

        <motion.button
          variants={fadeInUp}
          onClick={scrollToDemo}
          className="group relative inline-flex items-center gap-2 px-8 py-4 bg-primary-600 text-white font-semibold rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          立即体验 Demo
          <ArrowDown className="w-5 h-5 group-hover:animate-bounce" />
        </motion.button>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <div className="w-6 h-10 border-2 border-gray-400 rounded-full flex justify-center pt-2">
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
            className="w-1.5 h-1.5 bg-gray-400 rounded-full"
          />
        </div>
      </motion.div>
    </section>
  );
}
