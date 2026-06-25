import { motion } from 'framer-motion';
import { 
  MessageSquare, 
  Search, 
  Brain, 
  FileText,
  GitBranch,
  Shield
} from 'lucide-react';

const features = [
  {
    icon: MessageSquare,
    title: '多轮对话',
    description: '基于 LangGraph 状态机管理上下文，支持复杂的多轮问诊对话',
    color: 'primary'
  },
  {
    icon: Search,
    title: '症状问诊',
    description: '通过意图识别判断用户意图，自动匹配推荐科室和可能疾病',
    color: 'medical'
  },
  {
    icon: Brain,
    title: '意图识别',
    description: '智能区分症状问诊、流程咨询、混合意图，精准路由到不同处理分支',
    color: 'purple'
  },
  {
    icon: FileText,
    title: 'RAG 检索',
    description: '结合 Elasticsearch 流程文档检索 + Milvus 向量知识库，提供精准答案',
    color: 'orange'
  },
  {
    icon: GitBranch,
    title: 'Agentic 编排',
    description: 'LangGraph 状态机编排多个节点，支持 Query 重写、文档评估等复杂逻辑',
    color: 'teal'
  },
  {
    icon: Shield,
    title: '会话管理',
    description: '基于 Redis 的多会话管理，支持会话切换、历史记录、状态持久化',
    color: 'rose'
  }
];

const colorClasses: Record<string, string> = {
  primary: 'bg-primary-100 text-primary-600',
  medical: 'bg-medical-100 text-medical-600',
  purple: 'bg-purple-100 text-purple-600',
  orange: 'bg-orange-100 text-orange-600',
  teal: 'bg-teal-100 text-teal-600',
  rose: 'bg-rose-100 text-rose-600'
};

const container = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

export function Features() {
  return (
    <section className="py-20 bg-gray-50">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-4">核心功能</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            六大核心能力，构建完整的智能导诊系统
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: "-50px" }}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={item}
              className="group bg-white rounded-2xl p-6 shadow-sm hover:shadow-lg transition-all duration-300 border border-gray-100 hover:border-primary-200"
              whileHover={{ y: -5 }}
            >
              <div className={`w-12 h-12 rounded-xl ${colorClasses[feature.color]} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
