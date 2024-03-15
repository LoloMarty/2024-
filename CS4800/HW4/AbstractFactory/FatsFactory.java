package AbstractFactory;

public class FatsFactory extends Nutrient {
    private String[] availableFats = { "Avocado", "Sour cream", "Tuna", "Peanuts" };

    public String getAllowedFood(String[] foodExceptions) {
        String allowedFood = "";
        Boolean pass = true;

        for (String fats : this.availableFats) {
            for (String food : foodExceptions) {
                if (fats.equalsIgnoreCase(food)) {
                    pass = false;
                }
            }

            if (pass == true) {
                allowedFood = fats;
            }

            pass = true;
        }

        return allowedFood;
    }

}
