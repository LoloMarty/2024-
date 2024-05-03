import java.util.Map;
import java.util.HashMap;

// Flyweight factory that creates and manages CharacterProperties objects
// Reuse CharacterProperties that have the same properties as other characters
class CharacterPropertiesFactory {
    private final Map<String, CharacterProperties> flyweights = new HashMap<>();

    // Add characterProperty to Map if not inside. If it is inside retrieve it without creating duplicate.
    public CharacterProperties setAndRetrieveFlyweightCharacterProperties(String font, String color, int size) {
        String key = font + size + color;
        if (!flyweights.containsKey(key)) {
            CharacterProperties properties = new CharacterProperties(font, color, size);
            flyweights.put(key, properties);
        }
        return flyweights.get(key);
    }

    // Not required - to show # of flyweight Objects
    public int sizeOfMap() {
        return flyweights.size();
    }

    // Deleting would require checking no other character in document uses the character property that is set for removal.
}
