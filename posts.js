// Central post registry for the rLLM site.
// Edit this list to add or update blog posts; index.html and blog.html render from it.

const POSTS = [
  {
    title: "Continual Learning via Real-Time RL for Agents",
    excerpt:
      "A study on real-time RL for continual agent learning, introducing Population-Relative Policy Optimization (PRPO).",
    date: "2026-06-22",
    authors: "Linbo Liu, Yuzhe Lu, Youzhi Luo, Panpan Xu, and the rLLM Team",
    readingTime: "17 min",
    category: "Research",
    link: "post.html?post=continual_learning.md",
    cover: "g-emerald",
    image: "assets/continual_learning/sync_async_training.png",
  },
  {
    title: "Hive: Collaborative Agent Evolution Platform",
    excerpt:
      "Hive is a collaborative platform for evolving and improving agents together. A swarm of agents iterate on shared tasks, learning from each other to push past what any single agent can reach alone.",
    date: "2026-03-18",
    authors: "The rLLM Team",
    readingTime: null,
    category: "Platform",
    link: "https://hive.rllm-project.com",
    external: true,
    cover: "g-violet",
  },
  {
    title: "rLLM UI: Real-Time Observability for Agent Training & Evaluation",
    excerpt:
      "A real-time observability platform for training and evaluating agents. Other tools show what is happening during training — rLLM UI shows you why, letting you inspect exactly what the model generates at every step.",
    date: "2026-03-16",
    authors: "Chanbin Park and the rLLM Team",
    readingTime: "7 min",
    category: "Release",
    link: "post.html?post=rllm_ui.md",
    cover: "g-cyan",
    image: "assets/rllm/sdk_arch.png",
  },
  {
    title: "On-Policy Distillation: Training Smaller Students from Stronger Teachers",
    excerpt:
      "rLLM On-Policy Distillation (OPD) trains smaller students from stronger teachers by using the teacher's policy to guide the student's training — a practical recipe for compact, capable models.",
    date: "2026-03-06",
    authors: "Brian Chen, Kyle Montgomery, and the rLLM Team",
    readingTime: "8 min",
    category: "Research",
    link: "post.html?post=opd.md",
    cover: "g-blue",
    image: "assets/opd/opd.png",
  },
  {
    title: "Faster and Better: Open-Source Recipe for Deep Research Agents",
    excerpt:
      "We achieve 5× faster training (1 day vs 5 days) for deep research agents with rLLM's fully asynchronous architecture, and push accuracy from 30% to 36% on BrowseComp-Plus with a simple test-time document cutoff.",
    date: "2026-02-19",
    authors: "rLLM Team",
    readingTime: "6 min",
    category: "Research",
    link: "post.html?post=deepresearch.md",
    cover: "g-indigo",
    image: "assets/deepresearch/browsecomp_plus_em.png",
  },
  {
    title: "rLLM-FinQA: A 4B Model that Outperforms 235B and Rivals Gemini 2.5 Pro",
    excerpt:
      "In a collaboration with Snorkel AI, a domain-specialized 4B model outperforms Qwen3-235B (59.7% vs 51.4%) and performs comparably to Gemini 2.5 Pro (60.6%) on an expert-curated agentic financial benchmark.",
    date: "2026-02-18",
    authors: "Manan Roongta, Sijun Tan, Bhavishya Pohani, Charles Dickens, Christopher Glaze",
    readingTime: "17 min",
    category: "Research",
    link: "post.html?post=finqa.md",
    cover: "g-teal",
    image: "assets/finqa/training_curve.png",
  },
  {
    title: "rLLM SDK: Training Any Agentic Program without Code Changes",
    excerpt:
      "The rLLM SDK intercepts LLM calls directly, letting you train any agent framework — LangChain, LangGraph, AutoGen, or custom code — without rewriting for training. What's trainable = what's practical to build.",
    date: "2025-12-10",
    authors: "Tianhao Wu, Sijun Tan, and the rLLM team",
    readingTime: "8 min",
    category: "Release",
    link: "post.html?post=sdk.md",
    cover: "g-amber",
    image: "assets/rllm/sdk_code_diff.png",
  },
  {
    title: "rLLM v0.2: RL Training over General Agentic Programs",
    excerpt:
      "A major upgrade introducing AgentWorkflowEngine and AgentWorkflowTrainer — general abstractions that let you define multi-agent systems and complex workflows, and train them with RL without rewriting production code.",
    date: "2025-10-16",
    authors: "Sijun Tan, Kyle Montgomery, and the rLLM team",
    readingTime: "10 min",
    category: "Release",
    link: "post.html?post=rllm_v0.2.md",
    cover: "g-rose",
    image: "assets/rllm/math_curve.png",
  },
  {
    title: "Pepper: An Event-Driven Architecture for Proactive Agentic Systems",
    excerpt:
      "Pepper is a real-time, event-driven architecture enabling proactive agentic systems. Our personal assistant proactively fetches and summarizes emails and provides context before you even start a conversation.",
    date: "2025-10-02",
    authors: "Tianhao Wu, Sijun Tan",
    readingTime: "13 min",
    category: "Research",
    link: "post.html?post=pepper.md",
    cover: "g-slate",
    image: "assets/pepper/pepper-arch.png",
  },
  {
    title: "rLLM: Reinforcement Learning for Language Agents",
    excerpt:
      "We release rLLM, an open-source framework for post-training language agents via reinforcement learning. Build custom agents and environments, train them with RL, and deploy them for real-world workloads.",
    date: "2025-07-01",
    authors: "Sijun Tan, Michael Luo, Colin Cai",
    readingTime: "10 min",
    category: "Release",
    link: "https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31",
    external: true,
    cover: "g-blue",
  },
];

function formatDate(iso) {
  const [y, m, d] = iso.split("-").map((n) => parseInt(n, 10));
  const dt = new Date(y, m - 1, d);
  return dt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function metaLine(post) {
  const parts = [formatDate(post.date)];
  if (post.readingTime) parts.push(`${post.readingTime} read`);
  return parts.join("  ·  ");
}

function coverMarkup(post, label) {
  const target = post.external ? ` target="_blank" rel="noopener noreferrer"` : "";
  if (post.image) {
    return `<a class="card-cover ${post.cover}" href="${post.link}"${target} aria-label="${post.title}">
        <img src="${post.image}" alt="" loading="lazy" />
      </a>`;
  }
  return `<a class="card-cover ${post.cover} card-cover--text" href="${post.link}"${target} aria-label="${post.title}">
      <span class="cover-mark">rLLM</span>
      <span class="cover-title">${label}</span>
    </a>`;
}

function cardMarkup(post) {
  const target = post.external ? ` target="_blank" rel="noopener noreferrer"` : "";
  const shortTitle = post.title.split(":")[0];
  return `<article class="post-card">
    ${coverMarkup(post, shortTitle)}
    <div class="post-card-body">
      <span class="tag tag--${post.category.toLowerCase()}">${post.category}</span>
      <h3 class="post-card-title"><a href="${post.link}"${target}>${post.title}</a></h3>
      <p class="post-card-excerpt">${post.excerpt}</p>
      <div class="post-card-meta">
        <span class="post-card-authors">${post.authors}</span>
        <span class="post-card-date">${metaLine(post)}</span>
      </div>
    </div>
  </article>`;
}

function featuredMarkup(post) {
  const target = post.external ? ` target="_blank" rel="noopener noreferrer"` : "";
  const shortTitle = post.title.split(":")[0];
  return `<article class="featured-card">
    ${coverMarkup(post, shortTitle)}
    <div class="featured-body">
      <span class="tag tag--${post.category.toLowerCase()}">${post.category}</span>
      <h2 class="featured-title"><a href="${post.link}"${target}>${post.title}</a></h2>
      <p class="featured-excerpt">${post.excerpt}</p>
      <div class="post-card-meta">
        <span class="post-card-authors">${post.authors}</span>
        <span class="post-card-date">${metaLine(post)}</span>
      </div>
    </div>
  </article>`;
}

function renderPosts({ featuredEl, gridEl, limit = null, withFeatured = false } = {}) {
  let list = POSTS.slice();
  let rest = list;
  if (withFeatured && featuredEl && list.length) {
    featuredEl.innerHTML = featuredMarkup(list[0]);
    rest = list.slice(1);
  }
  if (limit) rest = rest.slice(0, limit);
  if (gridEl) gridEl.innerHTML = rest.map(cardMarkup).join("");
}

// Allow Node tooling (e.g. generate_rss.js) to consume the post registry.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { POSTS };
}
