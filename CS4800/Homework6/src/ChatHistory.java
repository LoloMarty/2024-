class ChatHistory {
    String messages;

    public String getMessages()
    {
        return this.messages;
    }

    public void set(String username, Message givenMessage)
    {
        messages += "\n" + username + " says: " + givenMessage.getText();
    }

    public MessageMomento takeSnapshot()
    {
        return new MessageMomento(this.messages);
    }

    public void restore(MessageMomento momento)
    {
        this.messages = momento.getSavedText();
    }

    public class MessageMomento {
        private final String message;
        private MessageMomento(String messageToSave)
        {
            this.message = messageToSave;
        }

        public String getSavedText()
        {
            return this.message;
        }
    }
}
