package AbstractFactory;

public class Customer {
    private String name;
    private String dietType;
    private Plan dietPlan;

    public Customer(String name, String dietType) {
        this.name = name;
        this.dietType = dietType;
        this.dietPlan = new Plan();
    }

    public Meal getMeal() {
        Meal meal = this.dietPlan.getMeal(dietType);

        System.out.printf("\n\nName: %s", this.name);
        System.out.printf("\nDiet Type: %s", this.dietType);
        System.out.printf("\nCarb: %s", meal.getCarb());
        System.out.printf("\nProtein: %s", meal.getProtein());
        System.out.printf("\nFat: %s", meal.getFat());

        return meal;
    }
}
