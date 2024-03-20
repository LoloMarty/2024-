package AbstractFactory;

public class Plan {
    private String[] allergicFoods;
    private Meal meal;
    MacronutrientFactory macroFactory = MacronutrientFactory.getInstance();

    public Plan() {
        this.meal = new Meal();
    }

    public Meal getMeal(String dietType) {
        if (dietType.equals("Paleo")) {
            this.allergicFoods = new String[] { "Cheese", "Bread", "Lentils", "Tofu", "Cheese", "Sour cream" };

        } else if (dietType.equals("Vegan")) {
            this.allergicFoods = new String[] { "Fish", "Chicken", "Tuna", "Cheese", "Sour cream" };

        } else if (dietType.equals("Nut Allergy")) {
            this.allergicFoods = new String[] { "Pistachio", "Peanuts" };

        } else {
            this.allergicFoods = new String[] { "" };
        }

        meal.setCarb(macroFactory.getFactory("carbs").getAllowedFood(allergicFoods));
        meal.setProtein(macroFactory.getFactory("proteins").getAllowedFood(allergicFoods));
        meal.setFat(macroFactory.getFactory("fats").getAllowedFood(allergicFoods));

        return meal;
    }

}
