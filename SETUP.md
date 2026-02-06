# Idle Machines - Setup Complete ✅

Modern tech-aesthetic Jekyll blog with warm paper light theme and subtle ink dark theme.

## 🎨 Theme Features

### Color Palette

**Light Mode (Warm Paper)**
- Background: `#F7F3EC` - warm paper texture
- Cards: `#FAF7F1` - slightly lighter
- Text: `#1F2833` - deep slate
- Links: `#365C7A` - muted blue
- Code blocks: `#F1EBDD` - paper-toned

**Dark Mode (Ink)**
- Background: `#0E1111` - deep ink
- Cards: `#151A1E` - slightly lighter
- Text: `#E8E5E0` - soft white
- Links: `#9EB9D1` - light blue
- Code blocks: `#1A2228` - dark slate

### Typography
- **Body**: Inter (clean, modern sans-serif)
- **Code**: IBM Plex Mono (technical, readable)

## 📁 What's Been Set Up

### Configuration Files

- `_config.yml` - Updated with:
  - Site title: "Idle Machines"
  - Tagline: "notes on training and theory"
  - Math rendering enabled by default (`math: true`)
  - Kramdown with MathJax
  - Comments disabled by default

### Custom Styling

- `_sass/custom/variables.scss` - Theme color variables for light/dark modes
- `_sass/custom/custom.scss` - Custom component styling
- `assets/css/jekyll-theme-chirpy.scss` - Main stylesheet that imports everything
- `_includes/head/custom.html` - Google Fonts (Inter + IBM Plex Mono)

### Navigation Tabs

- **Archives** (order: 1) - All posts chronologically
- **Projects** (order: 2) - Curated project write-ups
- **Categories** (order: 3) - Posts by category
- **About** (order: 4) - Site information

### Sample Posts

✅ `_posts/2025-12-12-welcome-to-idle-machines.md` - Introduction post
✅ `_posts/2025-12-12-math-and-code-demo.md` - Demonstrates math rendering, code blocks, tables

### Draft Templates

📝 `_drafts/weekly-notes-template.md` - Reusable weekly notes structure
📝 `_drafts/rl-darts-environment.md` - RL environment project stub
📝 `_drafts/fp8-fp4-blackwell-experiments.md` - Low-precision training stub

### Utility Scripts

🛠️ `tools/new-weekly.sh` - Generate new weekly notes with one command

## 🚀 Quick Start

### Run Development Server

```bash
bundle exec jekyll serve
```

Site will be available at: http://127.0.0.1:4000

If port 4000 is busy:
```bash
bundle exec jekyll serve --port 4001
```

### Create New Weekly Notes

```bash
./tools/new-weekly.sh
```

This creates a new post in `_posts/` with:
- Current date
- Week range in title
- Pre-filled template sections

### Create New Post

```bash
# Manually create in _posts/
touch _posts/YYYY-MM-DD-title-here.md
```

Front matter template:
```yaml
---
title: "Your Title"
date: YYYY-MM-DD HH:MM:SS +0000
categories: [Category1, Category2]
tags: [tag1, tag2, tag3]
math: true        # Enable if using equations
mermaid: false    # Enable if using diagrams
---
```

### Publish a Draft

Move from `_drafts/` to `_posts/` and set proper date:

```bash
mv _drafts/my-post.md _posts/2025-12-15-my-post.md
```

Remove `published: false` from front matter.

## 📝 Writing Tips

### Math Rendering

Inline: `$E = mc^2$` renders as $E = mc^2$

Display block:
```latex
$$
\mathcal{L}(\theta) = \mathbb{E}_{x,y}[\ell(f_\theta(x), y)]
$$
```

### Code Blocks

````markdown
```python
def train(model, data):
    optimizer.zero_grad()
    loss = model(data)
    loss.backward()
    optimizer.step()
```
````

Supported languages: python, bash, yaml, json, javascript, rust, c++, etc.

### Tables

```markdown
| Method | Accuracy | Speed |
|--------|----------|-------|
| FP32   | 94.2%    | 1.0x  |
| FP16   | 94.1%    | 1.8x  |
| FP8    | 93.8%    | 3.2x  |
```

### Alerts/Callouts

```markdown
> **Note**: This is important information
{: .prompt-info }

> **Warning**: Be careful here
{: .prompt-warning }

> **Tip**: Pro tip for readers
{: .prompt-tip }
```

## 🎯 Planned Content Structure

### Weekly Notes
- Training observations
- Paper summaries
- Random technical thoughts
- Next week's tasks

### Projects
- RL Darts Environment
- FP8/FP4 Training on Blackwell
- Other experiments

### Categories
- Weekly Notes
- Projects
- Reinforcement Learning
- Low-Precision Training
- Theory
- Meta

## 🔧 Customization

### Change Colors

Edit `_sass/custom/variables.scss`:

```scss
:root {
  --body-bg: #YOUR_COLOR;
  --text-color: #YOUR_COLOR;
  // ... etc
}
```

### Change Fonts

Edit `_includes/head/custom.html`:

```html
<link href="https://fonts.googleapis.com/css2?family=YOUR_FONT&display=swap" rel="stylesheet">
<style>
  :root {
    --font-family: "YOUR_FONT", ...;
  }
</style>
```

### Modify Navigation

Edit files in `_tabs/`:
- Change `order:` to reorder tabs
- Change `icon:` to use different Font Awesome icons

## 📦 Dependencies

- Ruby 3.3.6 (via rbenv)
- Jekyll 4.4.1
- Chirpy theme 7.4.1
- Bundler 4.0.1

## 🌐 Deployment

This site is ready to deploy to GitHub Pages.

### Push to GitHub

```bash
git add .
git commit -m "Initial Idle Machines setup"
git push origin main
```

GitHub Pages will automatically build and deploy.

### Custom Domain (Optional)

Add `CNAME` file:
```
yourdomain.com
```

Update `_config.yml`:
```yaml
url: "https://yourdomain.com"
```

## 📊 Analytics (Optional)

To enable Google Analytics, edit `_config.yml`:

```yaml
analytics:
  google:
    id: G-XXXXXXXXXX
```

## ✨ What Makes This Different

1. **Warm aesthetic** - No harsh whites, gentle on the eyes
2. **Math-first** - LaTeX rendering enabled by default
3. **Code-friendly** - Syntax highlighting that matches the theme
4. **Frictionless writing** - Templates and scripts for quick posting
5. **Technical focus** - Optimized for ML/RL/training content

## 🎨 Design Philosophy

> Modern tech-aesthetic. Think Anthropic docs, Linear app, Notion—but not trying to ape them. Warm beige/paper color, soft grey text. Clean, minimal, focused on content.

The theme prioritizes:
- **Readability** - Generous whitespace, comfortable line lengths
- **Technical content** - Math, code, tables all first-class
- **Speed** - Fast load times, minimal JavaScript
- **Professionalism** - Subtle, not flashy

---

**Note**: The deprecation warnings about Sass `@import` are harmless—they're from the upstream Chirpy theme. The site builds and works perfectly.

Enjoy your new technical blog! 📝✨
