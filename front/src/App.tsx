import { Hero } from './components/Hero';
import { ProjectIntro } from './components/ProjectIntro';
import { Features } from './components/Features';
import { Architecture } from './components/Architecture';
import { TechStack } from './components/TechStack';
import { LiveDemo } from './components/LiveDemo';
import { Footer } from './components/Footer';

function App() {
  return (
    <div className="min-h-screen">
      <Hero />
      <ProjectIntro />
      <Features />
      <Architecture />
      <TechStack />
      <LiveDemo />
      <Footer />
    </div>
  );
}

export default App;
