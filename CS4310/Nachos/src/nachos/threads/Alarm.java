package nachos.threads;

import java.util.PriorityQueue;

import nachos.machine.*;

/**
 * Uses the hardware timer to provide preemption, and to allow threads to sleep
 * until a certain time.
 */
public class Alarm {

    // we will not use a condition variable. We need access to the threads
    // associated time.
    // likewise for communicator.

    private class ThreadToBeAwoken implements Comparable<ThreadToBeAwoken> {
        Lock lock = new Lock();
        Condition condition = new Condition(lock);
        public long wakeTime;

        @Override
        public int compareTo(ThreadToBeAwoken o) {
            if (this.wakeTime < o.wakeTime) {
                return -1;
            } else if (this.wakeTime > o.wakeTime) {
                return 1;
            }
            return 0;

        }
    }

    /**
     * Allocate a new Alarm. Set the machine's timer interrupt handler to this
     * alarm's callback.
     *
     * <p>
     * <b>Note</b>: Nachos will not function correctly with more than one
     * alarm.
     */
    public Alarm() {
        Machine.timer().setInterruptHandler(new Runnable() {
            public void run() {
                timerInterrupt();
            }
        });
    }

    /**
     * The timer interrupt handler. This is called by the machine's timer
     * periodically (approximately every 500 clock ticks). Causes the current
     * thread to yield, forcing a context switch if there is another thread
     * that should be run.
     */
    public void timerInterrupt() {
        if (threadsToBeAwokenQueue.isEmpty()) {
            return;
        } else {
            long currentTime = Machine.timer().getTime();
            while (!threadsToBeAwokenQueue.isEmpty() && threadsToBeAwokenQueue.peek().wakeTime <= currentTime) {
                ThreadToBeAwoken thread = threadsToBeAwokenQueue.poll();
                boolean intStatus = Machine.interrupt().disable();
                thread.lock.acquire();
                thread.condition.wake();
                thread.lock.release();
                Machine.interrupt().restore(intStatus);
            }
        }
    }

    /**
     * Put the current thread to sleep for at least <i>x</i> ticks,
     * waking it up in the timer interrupt handler. The thread must be
     * woken up (placed in the scheduler ready set) during the first timer
     * interrupt where
     *
     * <p>
     * <blockquote>
     * (current time) >= (WaitUntil called time)+(x)
     * </blockquote>
     *
     * @param x the minimum number of clock ticks to wait.
     *
     * @see nachos.machine.Timer#getTime()
     */
    public void waitUntil(long x) {
        long wakeTime = Machine.timer().getTime() + x;
        ThreadToBeAwoken threadToBeAwoken = new ThreadToBeAwoken();
        threadToBeAwoken.wakeTime = wakeTime;
        threadsToBeAwokenQueue.add(threadToBeAwoken);
        boolean intStatus = Machine.interrupt().disable();
        threadToBeAwoken.lock.acquire();
        threadToBeAwoken.condition.sleep();
        threadToBeAwoken.lock.release();
        Machine.interrupt().restore(intStatus);
    }

    private PriorityQueue<ThreadToBeAwoken> threadsToBeAwokenQueue = new PriorityQueue<ThreadToBeAwoken>();
}
