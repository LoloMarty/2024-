package AbstractFactory;

public abstract class DietPlan {
    private String[] unacceptableFoods;

    public abstract String[] setUnacceptableFoods();

    public abstract String[] getUnacceptableFoods();
}
