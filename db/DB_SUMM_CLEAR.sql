/* Run this as root: $mysql -u root -p < DB_SUMM_CLEAR.sql */
USE STIR_CME_CL_SUMM;

DROP PROCEDURE IF EXISTS SelectAll;

DELIMITER $$
CREATE PROCEDURE SelectAll()
BEGIN
  SHOW tables;
END$$