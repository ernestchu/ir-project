# Lyric Search

Depoly at https://ernestchu.github.io/ir-project/  
Backend: https://huggingface.co/spaces/ernestchu/lyric-search

## TODO
- [x] longest common substring
- [x] Scroll to top after search
- [x] add clear search bar button
- [ ] term frequency upper bound
- [ ] Randomly sample 10,000 songs. For each song, randomly select a snippet of lyrics. Then, use our app to search with the snippet and evaluate whether the correct song appears within the top-k ranked search results.

## Setup

Make sure to install dependencies:

```bash
# npm
npm install

# pnpm
pnpm install

# yarn
yarn install

# bun
bun install
```

## Development Server

Start the development server on `http://localhost:3000`:

```bash
# npm
npm run dev

# pnpm
pnpm dev

# yarn
yarn dev

# bun
bun run dev
```

## Production

Build the application for production:

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build

# bun
bun run build
```

Locally preview production build:

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview

# bun
bun run preview
```

Check out the [deployment documentation](https://nuxt.com/docs/getting-started/deployment) for more information.
