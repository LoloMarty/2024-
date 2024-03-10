package CS4800.Composition;

public class Folder {
    private Folder[] subFolders;
    private File[] files;
    private String name;

    public Folder() {
        this.subFolders = new Folder[0];
        this.files = new File[0];
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Folder[] getSubFolders() {
        return subFolders;
    }

    public void setSubFolders(Folder[] subFolders) {
        this.subFolders = subFolders;
    }

    public File[] getFiles() {
        return files;
    }

    public void setFiles(File[] files) {
        this.files = files;
    }

    private File makeNewFile(String givenFileName) {
        return new File(givenFileName);
    }

    private Folder makeNewFolder() {
        return new Folder();
    }

    private int getLastIndex(File[] givenFileGroup) {
        return givenFileGroup.length - 1;
    }

    private int getLastIndex(Folder[] givenFolderGroup) {
        return givenFolderGroup.length - 1;
    }

    public void addFile(String givenFileNameString) {
        File[] newFilesGroup = fileGroupSizePlusOne();
        newFilesGroup = this.copyFiles(this.getFiles());
        newFilesGroup[getLastIndex(newFilesGroup)] = makeNewFile(givenFileNameString);
        this.setFiles(newFilesGroup);
    }

    public void addFolder(String givenFolderName) {
        Folder[] newSubFoldersGroup = subFolderGroupSizePlusOne();
        newSubFoldersGroup = this.copyFolders(this.getSubFolders());

        Folder newFolder = makeNewFolder();
        newFolder.setName(givenFolderName);

        newSubFoldersGroup[getLastIndex(newSubFoldersGroup)] = newFolder;

        this.setSubFolders(newSubFoldersGroup);
    }

    public void addFolder(Folder givenFolder) {
        Folder[] newSubFoldersGroup = subFolderGroupSizePlusOne();
        newSubFoldersGroup = this.copyFolders(this.getSubFolders());

        newSubFoldersGroup[getLastIndex(newSubFoldersGroup)] = givenFolder;

        this.setSubFolders(newSubFoldersGroup);
    }

    private Folder[] subFolderGroupSizePlusOne() {
        return new Folder[this.subFolders.length + 1];
    }

    private File[] fileGroupSizePlusOne() {
        return new File[this.files.length + 1];
    }

    private Folder[] copyFolders(Folder[] foldersToCopy) {
        Folder[] newFoldersGroupToReturn = new Folder[foldersToCopy.length + 1];

        for (int file = 0; file < foldersToCopy.length; file++) {
            newFoldersGroupToReturn[file] = foldersToCopy[file];
        }

        return newFoldersGroupToReturn;
    }

    private File[] copyFiles(File[] filesToCopy) {
        File[] newFilesGroupToReturn = new File[filesToCopy.length + 1];

        for (int file = 0; file < filesToCopy.length; file++) {
            newFilesGroupToReturn[file] = filesToCopy[file];
        }

        return newFilesGroupToReturn;
    }

    private void printIndents(int numberOfPrintedIndents) {
        for (int indent = 0; indent < numberOfPrintedIndents; indent++) {
            System.out.printf("   ");
        }
    }

    private void printNamesOfAllFilesInCurrentFolder(int numberOfPrintedIndents) {
        for (int file = 0; file < this.files.length; file++) {
            if (this.files[file] != null) {
                printIndents(numberOfPrintedIndents);
                this.files[file].printNameOfFile(numberOfPrintedIndents + 1);
            }
        }
    }

    private void printAllSubFoldersInCurrentFolder(int numberOfPrintedIndents) {
        for (int folder = 0; folder < this.subFolders.length; folder++) {
            if (this.subFolders[folder] != null) {
                printIndents(numberOfPrintedIndents);
                this.subFolders[folder].printEntireDirectory_helper(numberOfPrintedIndents + 1);
            }
        }
    }

    public void printEntireDirectory_helper(int numberOfPrintedIndents) {
        printIndents(numberOfPrintedIndents);
        System.out.printf("[Folder] %s\n", this.getName());

        printAllSubFoldersInCurrentFolder(numberOfPrintedIndents);
        printNamesOfAllFilesInCurrentFolder(numberOfPrintedIndents);

    }

    public void printEntireDirectory() {
        System.out.printf("[Folder] %s\n", this.getName());

        printAllSubFoldersInCurrentFolder(0);
        printNamesOfAllFilesInCurrentFolder(0);

    }

    private int indexOfFoldertWithName(String nameOfSearchingFolder) {
        int searchIndex = -1;

        for (int folder = 0; folder < this.subFolders.length - 1; folder++) {
            if (this.subFolders[folder] != null) {
                if (this.subFolders[folder].getName() == nameOfSearchingFolder) {
                    searchIndex = folder;
                }
            }
        }

        for (int subFolder = 0; subFolder < this.subFolders.length - 1; subFolder++) {
            if (this.subFolders[subFolder] != null) {
                this.subFolders[subFolder].removeFolder(nameOfSearchingFolder);
            }
        }

        return searchIndex;
    }

    public void removeFolder(String nameOfFolderToRemove) {
        int searchResult = indexOfFoldertWithName(nameOfFolderToRemove);

        if (searchResult != -1) {
            this.subFolders[searchResult] = null;
        }

    }
}