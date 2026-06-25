import { motion } from 'framer-motion';
import { 
  Globe, 
  Server, 
  Database, 
  Layers,
  Search
} from 'lucide-react';

const architectureData = [
  {
    layer: '前端层',
    icon: Globe,
    items: ['React + Vite', 'Tailwind CSS', 'Framer Motion', 'Rich CLI'],
    color: 'primary'
  },
  {
    layer: 'API 层',
    icon: Server,
    items: ['FastAPI', 'RESTful API', 'Pydantic 验证', 'CORS 配置'],
    color: 'blue'
  },
  {
    layer: '编排层',
    icon: Layers,
    items: ['LangGraph', '状态机', '意图路由', '节点决策'],
    color: 'purple'
  },
  {
    layer: '检索层',
    icon: Search,
    items: ['ES 流程检索', 'Milvus 向量检索', 'DashScope Embedding', 'Query 重写'],
    color: 'orange'
  },
  {
    layer: '基础设施',
    icon: Database,
    items: ['Redis 会话', 'ES 文档库', 'Milvus 向量库', 'DashScope LLM'],
    color: 'teal'
  }
];

const colorClasses: Record<string, { bg: string; border: string; icon: string }> = {
  primary: { bg: 'bg-primary-50', border: 'border-primary-200', icon: 'text-primary-600' },
  blue: { bg: 'bg-blue-50', border: 'border-blue-200', icon: 'text-blue-600' },
  purple: { bg: 'bg-purple-50', border: 'border-purple-200', icon: 'text-purple-600' },
  orange: { bg: 'bg-orange-50', border: 'border-orange-200', icon: 'text-orange-600' },
  teal: { bg: 'bg-teal-50', border: 'border-teal-200', icon: 'text-teal-600' }
};

const container = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.15
    }
  }
};

const item = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
  transition: { duration: 0.5 }
};

export function Architecture() {
  return (
    <section className="py-20 bg-white">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-4">系统架构</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            五层架构设计，从前端展示到后端检索，完整的数据流向
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="relative"
        >
          {/* 连接线 */}
          <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary-300 via-purple-300 to-teal-300 transform -translate-x-1/2"></div>

          {/* 移动端/桌面端显示 */}
          <div className="grid lg:grid-cols-2 gap-8">
            {architectureData.map((layer, index) => (
              <motion.div
                key={index}
                variants={item}
                className={`relative ${index % 2 === 0 ? 'lg:pr-12' : 'lg:pl-12'}`}
              >
                <div className={`flex items-start gap-4 p-6 rounded-2xl ${colorClasses[layer.color].bg} border ${colorClasses[layer.color].border}`}>
                  <div className={`p-3 rounded-xl bg-white shadow-sm ${colorClasses[layer.color].icon}`}>
                    <layer.icon className="w-6 h-6" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">{layer.layer}</h3>
                    <div className="flex flex-wrap gap-2">
                      {layer.items.map((tech, i) => (
                        <span 
                          key={i}
                          className="px-3 py-1 bg-white rounded-full text-sm font-medium text-gray-700 shadow-sm"
                        >
                          {tech}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                {/* 箭头指示 */}
                {index < architectureData.length - 1 && (
                  <div className={`lg:absolute ${index % 2 === 0 ? 'lg:right-0 lg:translate-x-1/2' : 'lg:left-0 lg:-translate-x-1/2'} top-1/2 -translate-y-1/2 lg:translate-y-0 mt-4 lg:mt-0`}>
                    <svg className={`w-6 h-6 ${colorClasses[layer.color].icon} mx-auto lg:mx-0`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
