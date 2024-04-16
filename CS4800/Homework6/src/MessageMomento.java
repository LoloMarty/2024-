import java.util.LinkedList;

public class MessageMomento {
    private final Message storedMessage;

    public MessageMomento(Message givenMessage)
    {
        this.storedMessage = givenMessage;
    }

    public Message getChatVersion()
    {
        return this.storedMessage;
    }
}
