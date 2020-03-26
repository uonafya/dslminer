import psycopg2
import click
from flask import current_app, g
from flask.cli import with_appcontext
import logging
log = logging.getLogger("db.py")

class database:

    def get_db_con(self):
        try:

            self.connection = psycopg2.connect(user = "postgres",
                                          password = "",
                                          host = "localhost",
                                          port = "5432",
                                          database = "mohdsl")
            connection =self.connection
            self.cursor = connection.cursor()
            log.info("Connected to database ")

        except (Exception, psycopg2.Error) as error :
            log.error ("Error while connecting to PostgreSQL", error)

        return (connection, self.cursor)


    def close_db_con(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()
            log.info("PostgreSQL connection is closed")
