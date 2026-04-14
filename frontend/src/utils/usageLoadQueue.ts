/**
 * Usage request queue that throttles API calls by group.
 *
 * Accounts sharing the same upstream (platform + type + proxy) are placed
 * into a single serial queue with a configurable delay between requests,
 * preventing upstream 429 rate-limit errors.
 *
 * Different groups run in parallel since they hit different upstreams.
 */

const GROUP_DELAY_MIN_MS = 1000
const GROUP_DELAY_MAX_MS = 1500

type Task<T> = {
  fn: () => Promise<T>
  resolve: (value: T) => void
  reject: (reason: unknown) => void
}

const queues = new Map<string, Task<unknown>[]>()
const running = new Set<string>()

function buildGroupKey(platform: string, type: string, proxyId: number | null): string {
  return `${platform}:${type}:${proxyId ?? 'direct'}`
}

async function drain(groupKey: string) {
  if (running.has(groupKey)) return
  running.add(groupKey)

  const queue = queues.get(groupKey)
  while (queue && queue.length > 0) {
    const task = queue.shift()!
    try {
      const result = await task.fn()
      task.resolve(result)
    } catch (err) {
      task.reject(err)
    }
    // Wait a random 1–1.5s before next request in the same group
    if (queue.length > 0) {
      const jitter = GROUP_DELAY_MIN_MS + Math.random() * (GROUP_DELAY_MAX_MS - GROUP_DELAY_MIN_MS)
      await new Promise((r) => setTimeout(r, jitter))
    }
  }

  running.delete(groupKey)
  queues.delete(groupKey)
}

/**
 * Enqueue a usage fetch call. Returns a promise that resolves when the
 * request completes (after waiting its turn in the group queue).
 */
export function enqueueUsageRequest<T>(
  platform: string,
  type: string,
  proxyId: number | null,
  fn: () => Promise<T>
): Promise<T> {
  const key = buildGroupKey(platform, type, proxyId)

  return new Promise<T>((resolve, reject) => {
    let queue = queues.get(key)
    if (!queue) {
      queue = []
      queues.set(key, queue)
    }
    queue.push({ fn, resolve, reject } as Task<unknown>)
    drain(key)
  })
}
