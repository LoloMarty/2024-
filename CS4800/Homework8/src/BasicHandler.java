public abstract class BasicHandler {
    private BasicHandler next;

    public BasicHandler(Snack givenSnack, BasicHandler next)
    {
        this.next = next;
    }

    public Snack handleRequest(String requestType)
    {
        Snack toReturn = null;
        if(next != null)
        {
            toReturn = next.handleRequest(requestType);
        }

        return toReturn;
    }
}
