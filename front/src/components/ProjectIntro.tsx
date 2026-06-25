import { motion } from 'framer-motion';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

export function ProjectIntro() {
  return (
    <section className="py-20 bg-white">
      <div className="max-w-4xl mx-auto px-4">
        <motion.div
          variants={fadeInUp}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: "-100px" }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-6">项目简介</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="p-6 bg-gradient-to-br from-primary-50 to-white rounded-2xl shadow-sm border border-primary-100">
              <h3 className="text-lg font-semibold text-primary-700 mb-3">🤔 解决什么问题</h3>
              <p className="text-gray-600 leading-relaxed">
                医院导诊台经常人满为患，患者需要快速了解挂号科室、就医流程、检查注意事项等信息。本系统通过 AI 对话的方式，7×24 小时为患者提供智能导诊服务。
              </p>
            </div>
            <div className="p-6 bg-gradient-to-br from-medical-50 to-white rounded-2xl shadow-sm border border-medical-100">
              <h3 className="text-lg font-semibold text-medical-600 mb-3">✨ 核心能力</h3>
              <p className="text-gray-600 leading-relaxed">
                支持症状问诊（判断可能疾病和推荐科室）、就医流程咨询（挂号、检查、取药等）、多轮对话上下文理解、基于 RAG 的专业医疗知识检索。
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
