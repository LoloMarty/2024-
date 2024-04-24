import java.util.HashMap;
import java.util.Map;

class CharacterPropertiesFactory {
    private static Map<String, CharacterProperties> propertiesMap = new HashMap<>();

    public static CharacterProperties getCharacterProperties(String font, String color, int size) {
        String key = font + color + size;
        if (!propertiesMap.containsKey(key)) {
            propertiesMap.put(key, new ConcreteCharacterProperties(font, color, size));
        }
        return propertiesMap.get(key);
    }
}