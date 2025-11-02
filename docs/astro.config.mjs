import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: "https://www.zhubert.com",
  base: import.meta.env.PROD ? "/transformer" : "/",
  trailingSlash: "always",
  integrations: [
    starlight({
      title: "Building a Transformer",
      description:
        "An educational journey through the architecture that powers modern AI",
      social: [
        {
          icon: "github",
          label: "Github",
          href: "https://github.com/zhubert/transformer",
        },
      ],
      sidebar: [
        {
          label: "Introduction",
          items: [{ label: "What is a Transformer?", link: "/" }],
        },
        {
          label: "Core Components",
          items: [
            {
              label: "Token Embeddings & Position Encoding",
              link: "/embeddings/",
            },
            { label: "Scaled Dot-Product Attention", link: "/attention/" },
            { label: "Multi-Head Attention", link: "/multi-head/" },
            { label: "Feed-Forward Networks", link: "/feedforward/" },
            { label: "Transformer Block", link: "/transformer-block/" },
            { label: "Complete Model", link: "/complete-model/" },
          ],
        },
        {
          label: "Advanced Topics",
          items: [
            { label: "Training at Scale", link: "/training/" },
            { label: "KV-Cache Optimization", link: "/kv-cache/" },
            { label: "Model Interpretability", link: "/interpretability/" },
          ],
        },
        {
          label: "Getting Started",
          items: [{ label: "Try It Yourself", link: "/try-it/" }],
        },
      ],
      customCss: ["./src/styles/custom.css"],
    }),
  ],
});
