# Transformer Documentation

This directory contains the Starlight-based documentation site for the Transformer educational project.

## Setup

1. Install dependencies:

```bash
cd docs
npm install
```

2. Start development server:

```bash
npm run dev
```

The site will be available at `http://localhost:4321/`

## Building for Production

```bash
npm run build
```

The built site will be in `dist/`.

## Preview Production Build

```bash
npm run preview
```

## Project Structure

```
docs/
├── src/
│   ├── content/
│   │   ├── docs/           # Documentation pages (MDX)
│   │   └── config.ts       # Content collection config
│   ├── styles/
│   │   └── custom.css      # Custom styling
│   └── env.d.ts            # TypeScript types
├── public/
│   └── assets/             # Images, diagrams, etc.
├── astro.config.mjs        # Astro + Starlight configuration
├── package.json            # Dependencies
└── tsconfig.json           # TypeScript configuration
```

## Adding New Pages

1. Create a new `.mdx` file in `src/content/docs/`
2. Add frontmatter with title and description:

```mdx
---
title: Page Title
description: Page description
---

Content here...
```

3. Update the sidebar in `astro.config.mjs` to include the new page

## Components Available

Starlight provides built-in components you can use:

```mdx
import { Aside, Card, CardGrid } from '@astrojs/starlight/components';

<Aside type="tip" title="Pro Tip">
  Helpful information
</Aside>

<Card title="Feature" icon="star">
  Feature description
</Card>
```

## Custom Styling

Add custom CSS to `src/styles/custom.css`. It's already imported in the Astro config.

## Learn More

- [Starlight Documentation](https://starlight.astro.build/)
- [Astro Documentation](https://docs.astro.build/)
