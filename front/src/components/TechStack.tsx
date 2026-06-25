import { motion } from 'framer-motion';
import { 
  Database, 
  Server, 
  Code, 
  Box, 
  Cloud,
  Terminal
} from 'lucide-react';

const techStacks = [
  { name: 'FastAPI', icon: Server, category: '后端框架', color: 'from-green-400 to-green-600' },
  { name: 'LangGraph', icon: Box, category: 'Agent 编排', color: 'from-blue-400 to-blue-600' },
  { name: 'Redis', icon: Database, category: '缓存与会话', color: 'from-red-400 to-red-600' },
  { name: 'Elasticsearch', icon: Database, category: '文档检索', color: 'from-yellow-400 to-yellow-600' },
  { name: 'Milvus', icon: Database, category: '向量检索', color: 'from-purple-400 to-purple-600' },
  { name: 'DashScope', icon: Cloud, category: 'LLM 服务', color: 'from-orange-400 to-orange-600' },
  { name: 'React', icon: Code, category: '前端框架', color: 'from-cyan-400 to-cyan-600' },
  { name: 'Tailwind CSS', icon: Code, category: '样式框架', color: 'from-sky-400 to-sky-600' },
  { name: 'Framer Motion', icon: Terminal, category: '动画库', color: 'from-pink-400 to-pink-600' },
];

const container = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.08
    }
  }
};

const item = {
  initial: { opacity: 0, scale: 0.8 },
  animate: { opacity: 1, scale: 1 },
  transition: { duration: 0.4 }
};

export function TechStack() {
  return (
    <section className="py-20 bg-gray-900 text-white">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold mb-4">技术栈</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            生产级技术选型，构建稳定可靠的智能导诊系统
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-6"
        >
          {techStacks.map((tech, index) => (
            <motion.div
              key={index}
              variants={item}
              className="group relative p-6 rounded-2xl bg-gray-800 border border-gray-700 hover:border-gray-600 transition-all duration-300"
              whileHover={{ scale: 1.05, y: -5 }}
            >
              <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${tech.color} opacity-0 group-hover:opacity-10 transition-opacity`}></div>
              <div className="relative flex flex-col items-center">
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${tech.color} flex items-center justify-center mb-3 shadow-lg group-hover:scale-110 transition-transform`}>
                  <tech.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-lg font-semibold mb-1">{tech.name}</h3>
                <p className="text-sm text-gray-400">{tech.category}</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
