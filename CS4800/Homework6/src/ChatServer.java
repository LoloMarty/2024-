import java.util.Deque;
import java.util.LinkedList;

public class ChatServer {
    private Deque<ChatHistory.MessageMomento> stateHistory;
    private ChatHistory chatHistory;

    public ChatServer()
    {
        this.stateHistory = new LinkedList<>();
        this.chatHistory = new ChatHistory();
    }

    public void sendMessage(String givenUsername, Message givenMessage)
    {
        this.chatHistory.set(givenUsername, givenMessage);
        this.stateHistory.add(this.chatHistory.takeSnapshot());
    }
    public void undo()
    {
        this.chatHistory.restore(this.stateHistory.pop());
    }

    public void getMessages()
}
