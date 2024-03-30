import java.util.Hashtable;

public abstract class FoodBase implements IFood{
    protected int basePrice;
    protected String foodName;
    protected String[] addedToppings;
    private final IFood wrapped;

    public FoodBase(IFood givenWrapped)
    {
        this.wrapped = givenWrapped;
    }

    public Integer getToppingPrice(String topping)
    {
        return Toppings.getInstance().getToppingPrice(topping);
    }

    @Override
    public int calculateCost()
    {
        return wrapped.calculateCost();
    }

}
