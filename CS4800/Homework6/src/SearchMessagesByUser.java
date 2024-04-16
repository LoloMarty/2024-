import java.util.Iterator;
import java.util.LinkedList;

public class SearchMessagesByUser implements Iterator<MessageMomento>{

    private LinkedList<MessageMomento> collection;

    public SearchMessagesByUser(LinkedList<MessageMomento> givenCollection)
    {
        this.collection = givenCollection;
    }

    public LinkedList<MessageMomento> getCollection() {
        return collection;
    }

    public void setCollection(LinkedList<MessageMomento> collection) {
        this.collection = collection;
    }

    @Override
    public boolean hasNext() {
        LinkedList<MessageMomento> copyList = null;

        for(MessageMomento message : this.collection)
        {
            copyList.addFirst(message);
        }

        copyList.pop();

        if (copyList.getFirst() != null)
        {
            return false;
        }else{
            return true;
        }
    }

    @Override
    public MessageMomento next() {
        MessageMomento returnItem = null;

        if (this.hasNext())
        {
            MessageMomento cycledItem = this.collection.pop();
            this.collection.addLast(cycledItem);
            returnItem = this.collection.getFirst();
        }

        return returnItem;
    }
}
