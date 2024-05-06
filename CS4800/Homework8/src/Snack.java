public class Snack {
    private final String name;
    private final int price;
    private int quantity;

    public Snack(String givenName, int givenPrice, int givenQuantity)
    {
        this.name = givenName;
        this.price = givenPrice;
        this.quantity = givenQuantity;
    }

    public String getName() {
        return name;
    }

    public int getPrice() {
        return price;
    }

    public int getQuantity() {
        return quantity;
    }

    public void snackDispensed()
    {
        this.quantity -= 1;
    }
}
