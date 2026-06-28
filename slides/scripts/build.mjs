import { cpSync, mkdirSync, readdirSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const args = process.argv.slice(2)
if (args[0] === '--') args.shift()

const slidevBin = process.platform === 'win32' ? 'slidev.cmd' : 'slidev'
const build = spawnSync(slidevBin, ['build', ...args], {
  cwd: root,
  stdio: 'inherit',
})

if (build.status !== 0) {
  process.exit(build.status ?? 1)
}

const outDir = resolve(root, getOutDir(args))
const drawDir = resolve(root, 'draw')
const publicDrawDir = join(outDir, 'draw')

mkdirSync(publicDrawDir, { recursive: true })

for (const entry of readdirSync(drawDir, { withFileTypes: true })) {
  if (!entry.isFile() || !entry.name.endsWith('.excalidraw')) continue
  cpSync(join(drawDir, entry.name), join(publicDrawDir, basename(entry.name)))
}

function getOutDir(argv) {
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === '--out' || arg === '-o') return argv[i + 1] ?? 'dist'
    if (arg.startsWith('--out=')) return arg.slice('--out='.length)
    if (arg.startsWith('-o=')) return arg.slice('-o='.length)
  }
  return 'dist'
}
