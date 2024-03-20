package AbstractFactory;

public class MacronutrientFactory {
    private static MacronutrientFactory instance;
    private static CarbsFactory carbFactory;
    private static ProteinFactory proteinFactory;
    private static FatsFactory fatsFactory;

    private MacronutrientFactory() {

    }

    public static MacronutrientFactory getInstance() {
        if (instance == null) {
            instance = new MacronutrientFactory();
        }

        return instance;
    }

    public static Nutrient getFactory(String type) {
        if (type.equalsIgnoreCase("carbs")) {
            if (carbFactory == null) {
                carbFactory = new CarbsFactory();
            }
            return carbFactory;

        } else if (type.equalsIgnoreCase("proteins")) {
            if (proteinFactory == null) {
                proteinFactory = new ProteinFactory();
            }
            return proteinFactory;

        } else if (type.equalsIgnoreCase("fats")) {
            if (fatsFactory == null) {
                fatsFactory = new FatsFactory();
            }
            return fatsFactory;

        } else {
            return null;
        }
    }
}
