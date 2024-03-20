package nachos.threads;

import java.util.LinkedList;

import nachos.machine.*;

/**
 * A <i>communicator</i> allows threads to synchronously exchange 32-bit
 * messages. Multiple threads can be waiting to <i>speak</i>,
 * and multiple threads can be waiting to <i>listen</i>. But there should never
 * be a time when both a speaker and a listener are waiting, because the two
 * threads can be paired off at this point.
 */
public class Communicator {
    /**
     * Allocate a new communicator.
     */
    public Communicator() {

    }

    private class ThreadToPerformCommunication {
        Lock lock;
        final Condition condition;
        boolean speaker;
        int message;

        public ThreadToPerformCommunication(int word) {
            this.message = word;
            this.lock = new Lock();
            condition = new Condition(lock);
        }

    }

    /**
     * Wait for a thread to listen through this communicator, and then transfer
     * <i>word</i> to the listener.
     *
     * <p>
     * Does not return until this thread is paired up with a listening thread.
     * Exactly one listener should receive <i>word</i>.
     *
     * @param word the integer to transfer.
     */
    public void speak(int word) {
        this.lock.acquire();
        if (!listeners.isEmpty()) {
            ThreadToPerformCommunication listener = listeners.remove();
            listener.message = word;

            listener.lock.acquire();
            listener.condition.wake();
            listener.lock.release();
        } else {
            ThreadToPerformCommunication speaker = new ThreadToPerformCommunication(word);
            speaker.lock.acquire();
            speakers.add(speaker);
            speaker.condition.sleep();
            this.lock.acquire();
            speaker.lock.release();
        }
        this.lock.release();
    }

    /**
     * Wait for a thread to speak through this communicator, and then return
     * the <i>word</i> that thread passed to <tt>speak()</tt>.
     *
     * @return the integer transferred.
     */
    public int listen() {
        this.lock.acquire();
        int word;
        if (!speakers.isEmpty()) {
            ThreadToPerformCommunication speaker = speakers.remove();
            word = speaker.message;

            speaker.lock.acquire();
            speaker.condition.wake();
            speaker.lock.release();
        } else {
            ThreadToPerformCommunication listener = new ThreadToPerformCommunication(0);
            listener.lock.acquire();
            listeners.add(listener);
            listener.condition.sleep();
            this.lock.acquire();
            word = listener.message;
            listener.lock.release();
        }
        this.lock.release();
        return word;
    }

    private Lock lock = new Lock();
    LinkedList<ThreadToPerformCommunication> speakers = new LinkedList<ThreadToPerformCommunication>();
    LinkedList<ThreadToPerformCommunication> listeners = new LinkedList<ThreadToPerformCommunication>();

}
