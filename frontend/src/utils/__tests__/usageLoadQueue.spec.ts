import { describe, expect, it, vi } from 'vitest'
import { enqueueUsageRequest } from '../usageLoadQueue'

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

describe('usageLoadQueue', () => {
  it('同组请求串行执行，间隔 >= 1s', async () => {
    const timestamps: number[] = []
    const makeFn = () => async () => {
      timestamps.push(Date.now())
      return 'ok'
    }

    const p1 = enqueueUsageRequest('anthropic', 'oauth', 1, makeFn())
    const p2 = enqueueUsageRequest('anthropic', 'oauth', 1, makeFn())
    const p3 = enqueueUsageRequest('anthropic', 'oauth', 1, makeFn())

    await Promise.all([p1, p2, p3])

    expect(timestamps).toHaveLength(3)
    // 随机 1-1.5s 间隔，至少 950ms（留一点误差）
    expect(timestamps[1] - timestamps[0]).toBeGreaterThanOrEqual(950)
    expect(timestamps[1] - timestamps[0]).toBeLessThan(1600)
    expect(timestamps[2] - timestamps[1]).toBeGreaterThanOrEqual(950)
    expect(timestamps[2] - timestamps[1]).toBeLessThan(1600)
  })

  it('不同组请求并行执行', async () => {
    const timestamps: Record<string, number> = {}
    const makeTracked = (key: string) => async () => {
      timestamps[key] = Date.now()
      return key
    }

    const p1 = enqueueUsageRequest('anthropic', 'oauth', 1, makeTracked('group1'))
    const p2 = enqueueUsageRequest('anthropic', 'oauth', 2, makeTracked('group2'))
    const p3 = enqueueUsageRequest('gemini', 'oauth', 1, makeTracked('group3'))

    await Promise.all([p1, p2, p3])

    // 不同组应几乎同时启动（差距 < 50ms）
    const values = Object.values(timestamps)
    const spread = Math.max(...values) - Math.min(...values)
    expect(spread).toBeLessThan(50)
  })

  it('请求失败时 reject，后续任务继续执行', async () => {
    const results: string[] = []

    const p1 = enqueueUsageRequest('anthropic', 'oauth', 99, async () => {
      throw new Error('fail')
    })
    const p2 = enqueueUsageRequest('anthropic', 'oauth', 99, async () => {
      results.push('second')
      return 'ok'
    })

    await expect(p1).rejects.toThrow('fail')
    await p2
    expect(results).toEqual(['second'])
  })

  it('返回值正确透传', async () => {
    const result = await enqueueUsageRequest('test', 'oauth', null, async () => {
      return { usage: 42 }
    })
    expect(result).toEqual({ usage: 42 })
  })

  it('proxy_id 为 null 的账号归为同一组', async () => {
    const order: number[] = []
    const makeFn = (n: number) => async () => {
      order.push(n)
      return n
    }

    const p1 = enqueueUsageRequest('anthropic', 'oauth', null, makeFn(1))
    const p2 = enqueueUsageRequest('anthropic', 'oauth', null, makeFn(2))

    await Promise.all([p1, p2])

    // 同组串行，按入队顺序执行
    expect(order).toEqual([1, 2])
  })
})
