package nachos.threads;

import java.util.LinkedList;

import nachos.machine.*;

/**
 * An implementation of condition variables that disables interrupt()s for
 * synchronization.
 *
 * <p>
 * You must implement this.
 *
 * @see	nachos.threads.Condition
 */
public class Condition2 {
    /**
     * Allocate a new condition variable.
     *
     * @param	conditionLock	the lock associated with this condition
     *				variable. The current thread must hold this
     *				lock whenever it uses <tt>sleep()</tt>,
     *				<tt>wake()</tt>, or <tt>wakeAll()</tt>.
     */
    public Condition2(Lock conditionLock) {
    	this.conditionLock = conditionLock;
    }

    /**
     * Atomically release the associated lock and go to sleep on this condition
     * variable until another thread wakes it using <tt>wake()</tt>. The
     * current thread must hold the associated lock. The thread will
     * automatically reacquire the lock before <tt>sleep()</tt> returns.
     */
    public void sleep() {
    	//check the current thread is holding the lock
		Lib.assertTrue(conditionLock.isHeldByCurrentThread());
	
		conditionLock.release();
		
		//record the interrupt's last state 
		boolean intStatus = Machine.interrupt().disable();
		//add the current thread to the wait queue
		waitQueue.add(KThread.currentThread());
		//put the thread to sleep, will wake later
		KThread.sleep();
		//restore interrupt capability
		Machine.interrupt().restore(intStatus);
		
		conditionLock.acquire();
    }

    /**
     * Wake up at most one thread sleeping on this condition variable. The
     * current thread must hold the associated lock.
     */
    public void wake() {
    	Lib.assertTrue(conditionLock.isHeldByCurrentThread());
	
    	if (waitQueue.isEmpty() == false){
    		//save interrupt status to restore later 
    		boolean intStatus = Machine.interrupt().disable();
    		//get first thread in queue
    		KThread thread = waitQueue.removeFirst();
    		
    		//need to check if the thread is null
    		if (thread != null){
    			thread.ready();
    		}
    		//restore interrupt status
    		Machine.interrupt().restore(intStatus);
    	}
    }

    /**
     * Wake up all threads sleeping on this condition variable. The current
     * thread must hold the associated lock.
     */
    public void wakeAll() {
		Lib.assertTrue(conditionLock.isHeldByCurrentThread());
		
		//iterate through all threads in the queue and wake them
		while(!waitQueue.isEmpty()){
			wake();
		}
    }

    private Lock conditionLock;
    private LinkedList<KThread> waitQueue;
}
