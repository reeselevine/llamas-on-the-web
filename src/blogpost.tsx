import { Fragment, ReactNode } from 'react';
import rawBlogpost from './content/blogpost.md?raw';

type BlogPostMeta = {
  title: string;
  byline: string[];
};

export type BlogNode =
  | { type: 'heading'; level: 2 | 3; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'rewrite'; id: string; text: string }
  | { type: 'demo' }
  | { type: 'benchmark' };

export type BlogPost = {
  meta: BlogPostMeta;
  nodes: BlogNode[];
};

const FRONTMATTER_KEYS = {
  title: 'title',
  byline: 'byline',
} as const;

const requiredMetaKeys = Object.values(FRONTMATTER_KEYS);

const parseFrontmatter = (
  source: string
): { frontmatter: Record<string, string>; body: string } => {
  if (!source.startsWith('---\n')) {
    throw new Error('Blogpost markdown is missing frontmatter.');
  }

  const closingIndex = source.indexOf('\n---\n', 4);
  if (closingIndex === -1) {
    throw new Error('Blogpost markdown frontmatter is not terminated.');
  }

  const frontmatterBlock = source.slice(4, closingIndex);
  const body = source.slice(closingIndex + 5).trim();
  const frontmatter = Object.fromEntries(
    frontmatterBlock
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        const separatorIndex = line.indexOf(':');
        if (separatorIndex === -1) {
          throw new Error(`Invalid frontmatter line: ${line}`);
        }
        const key = line.slice(0, separatorIndex).trim();
        const value = line.slice(separatorIndex + 1).trim();
        return [key, value];
      })
  );

  return { frontmatter, body };
};

const parseNodes = (body: string): BlogNode[] => {
  const lines = body.replace(/\r\n/g, '\n').split('\n');
  const nodes: BlogNode[] = [];
  let paragraphLines: string[] = [];
  let rewriteIndex = 0;

  const flushParagraph = () => {
    if (paragraphLines.length === 0) return;
    nodes.push({
      type: 'paragraph',
      text: paragraphLines.join(' ').trim(),
    });
    paragraphLines = [];
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      continue;
    }

    if (trimmed === ':::demo') {
      flushParagraph();
      nodes.push({ type: 'demo' });
      continue;
    }

    if (trimmed === ':::benchmark') {
      flushParagraph();
      nodes.push({ type: 'benchmark' });
      continue;
    }

    if (trimmed === ':::rewrite') {
      flushParagraph();
      const rewriteLines: string[] = [];

      while (i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        i += 1;
        if (nextLine.trim() === ':::') {
          break;
        }
        rewriteLines.push(nextLine.trim());
      }

      rewriteIndex += 1;
      nodes.push({
        type: 'rewrite',
        id: `rewrite-${rewriteIndex}`,
        text: rewriteLines.join(' ').trim(),
      });
      continue;
    }

    if (trimmed.startsWith('### ')) {
      flushParagraph();
      nodes.push({ type: 'heading', level: 3, text: trimmed.slice(4).trim() });
      continue;
    }

    if (trimmed.startsWith('## ')) {
      flushParagraph();
      nodes.push({ type: 'heading', level: 2, text: trimmed.slice(3).trim() });
      continue;
    }

    paragraphLines.push(trimmed);
  }

  flushParagraph();
  return nodes;
};

const parseBlogPost = (source: string): BlogPost => {
  const { frontmatter, body } = parseFrontmatter(source);
  const rawMeta = Object.fromEntries(
    Object.entries(FRONTMATTER_KEYS).map(([sourceKey, targetKey]) => [
      targetKey,
      frontmatter[sourceKey],
    ])
  );

  for (const key of requiredMetaKeys) {
    if (!rawMeta[key]) {
      throw new Error(`Blogpost frontmatter is missing "${key}".`);
    }
  }

  return {
    meta: {
      title: rawMeta.title,
      byline: rawMeta.byline.split('|').map((entry) => entry.trim()),
    },
    nodes: parseNodes(body),
  };
};

const ESCAPED_UNDERSCORE_PLACEHOLDER = '\uE000';

const restoreEscapedUnderscores = (text: string) =>
  text.replace(new RegExp(ESCAPED_UNDERSCORE_PLACEHOLDER, 'g'), '_');

const renderInlineMarkdown = (text: string): ReactNode[] => {
  const escapedText = text.replace(/\\_/g, ESCAPED_UNDERSCORE_PLACEHOLDER);
  const parts = escapedText.split(
    /(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_)/
  );

  return parts
    .filter((part) => part.length > 0)
    .map((part, index) => {
      if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
        return <code key={index}>{part.slice(1, -1)}</code>;
      }

      const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        return (
          <a
            key={index}
            href={linkMatch[2]}
            target="_blank"
            rel="noreferrer"
          >
            {restoreEscapedUnderscores(linkMatch[1])}
          </a>
        );
      }

      if (
        ((part.startsWith('**') && part.endsWith('**')) ||
          (part.startsWith('__') && part.endsWith('__'))) &&
        part.length >= 4
      ) {
        return (
          <strong key={index}>
            {restoreEscapedUnderscores(part.slice(2, -2))}
          </strong>
        );
      }

      if (
        ((part.startsWith('*') && part.endsWith('*')) ||
          (part.startsWith('_') && part.endsWith('_'))) &&
        part.length >= 3
      ) {
        return (
          <em key={index}>{restoreEscapedUnderscores(part.slice(1, -1))}</em>
        );
      }

      return <Fragment key={index}>{restoreEscapedUnderscores(part)}</Fragment>;
    });
};

export const blogPost = parseBlogPost(rawBlogpost);
export const blogPostPromptSource = parseFrontmatter(rawBlogpost).body
  .replace(/\n:::demo\n/g, '\n\n')
  .replace(/\n:::benchmark\n/g, '\n\n')
  .replace(/\n:::rewrite\n/g, '\n\n')
  .replace(/\n:::\n/g, '\n\n')
  .trim();
export { renderInlineMarkdown };
