import java.util.LinkedList;

class ChatHistory {
    private static LinkedList<Message> stateOfChat;
    private static LinkedList<MessageMomento> chatHistory;

    private static ChatHistory instance;

    public ChatHistory()
    {
        stateOfChat = new LinkedList<>();
        chatHistory = new LinkedList<>();
    }

    public static ChatHistory getInstance()
    {
        if(instance == null)
        {
            instance = new ChatHistory();
        }

        return instance;
    }

    public LinkedList<Message> getChatState()
    {
        return this.stateOfChat;
    }
    private LinkedList<Message> duplicateLinkedList(LinkedList<Message> originalList) {
        LinkedList<Message > duplicateList = new LinkedList<>();
        
        duplicateList.addAll(originalList);

        return duplicateList;
    }
    public void addMessage(Message givenMessage)
    {
        MessageMomento momento = new MessageMomento(this.duplicateLinkedList(stateOfChat));
        chatHistory.addFirst(momento);
        stateOfChat.add(givenMessage);
    }
    public void undo()
    {
        this.stateOfChat = chatHistory.pop().getChatVersion();
    }
    public class MessageMomento
    {
        private final LinkedList<Message> chatVersion;

        public MessageMomento(LinkedList<Message> givenChat)
        {
            this.chatVersion = givenChat;
        }

        public LinkedList<Message> getChatVersion()
        {
            return this.chatVersion;
        }
    }
}
