/*
 * test_vulnerable.c
 * Example C file with a classic buffer overflow vulnerability
 * Used for testing the vulnerability detection pipeline
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int process_input(char *user_input) {
    char buffer[64];
    strcpy(buffer, user_input);  /* Vulnerable: no bounds checking */
    printf("Processed: %s\n", buffer);
    return 0;
}

int read_file(char *filename) {
    FILE *fp;
    char line[128];
    fp = fopen(filename, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp)) {
        printf("%s", line);
    }
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input>\n", argv[0]);
        return 1;
    }
    process_input(argv[1]);
    return 0;
}
